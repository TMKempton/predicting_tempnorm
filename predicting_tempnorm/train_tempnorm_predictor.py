import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def infer_eval_lengths(records: List[Dict]) -> List[int]:
    cfg = records[0].get("config", {})
    lengths = cfg.get("eval_lengths")
    if not lengths:
        raise ValueError("Could not infer eval_lengths from first record config")
    return [int(x) for x in lengths]


def build_dataset(records: List[Dict], eval_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = f"individual_tempnorms_{eval_length}"

    x_rows: List[np.ndarray] = []
    y_vals: List[float] = []
    groups: List[int] = []

    for i, rec in enumerate(records):
        if key not in rec:
            raise KeyError(f"Missing key '{key}' in record {i}")

        hidden = np.asarray(rec["starting_hidden_vector"], dtype=np.float32)
        tempnorms = rec[key]
        group_id = int(rec.get("context_index", i))

        for val in tempnorms:
            x_rows.append(hidden)
            y_vals.append(float(val))
            groups.append(group_id)

    x = np.stack(x_rows, axis=0)
    y = np.asarray(y_vals, dtype=np.float64)
    g = np.asarray(groups, dtype=np.int64)

    return x, y, g


def train_eval_length_model(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
    ridge_alpha: float,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, np.ndarray]]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(x, y, groups=groups))

    x_train = x[train_idx]
    x_test = x[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Tempnorm values can be extremely small; fit in log-space for stability.
    y_train_log = np.log(np.clip(y_train, 1e-300, None))
    y_test_log = np.log(np.clip(y_test, 1e-300, None))

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=ridge_alpha)),
        ]
    )
    model.fit(x_train, y_train_log)

    pred_train_log = model.predict(x_train)
    pred_test_log = model.predict(x_test)

    pred_train = np.exp(pred_train_log)
    pred_test = np.exp(pred_test_log)

    metrics = {
        "train_mae_log": float(mean_absolute_error(y_train_log, pred_train_log)),
        "test_mae_log": float(mean_absolute_error(y_test_log, pred_test_log)),
        "train_rmse_log": float(np.sqrt(mean_squared_error(y_train_log, pred_train_log))),
        "test_rmse_log": float(np.sqrt(mean_squared_error(y_test_log, pred_test_log))),
        "test_r2_log": float(r2_score(y_test_log, pred_test_log)),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
    }

    split = {
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
    return model, metrics, split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hidden-vector -> tempnorm regressors")
    p.add_argument("--data-file", type=Path, required=True, help="JSONL created by run_experiment.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--eval-lengths", type=int, nargs="+", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.data_file)

    eval_lengths = args.eval_lengths if args.eval_lengths is not None else infer_eval_lengths(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: Dict[str, Dict[str, float]] = {}

    for eval_length in eval_lengths:
        x, y, groups = build_dataset(records, eval_length=eval_length)
        model, metrics, split = train_eval_length_model(
            x=x,
            y=y,
            groups=groups,
            test_size=args.test_size,
            seed=args.seed,
            ridge_alpha=args.ridge_alpha,
        )

        model_path = args.output_dir / f"tempnorm_model_eval_{eval_length}.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)

        split_path = args.output_dir / f"split_eval_{eval_length}.json"
        split_payload = {
            "train_idx": split["train_idx"].tolist(),
            "test_idx": split["test_idx"].tolist(),
        }
        split_path.write_text(json.dumps(split_payload), encoding="utf-8")

        all_metrics[str(eval_length)] = metrics

    metrics_payload = {
        "data_file": str(args.data_file),
        "eval_lengths": eval_lengths,
        "model": "standardized_ridge_log_target",
        "metrics": all_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()