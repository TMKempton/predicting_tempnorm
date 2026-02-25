import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

from .utils import read_jsonl


def predictive_analysis(
    rows: List[Dict],
    feature_lengths: List[int],
    target_length: int,
    test_size: float,
    seed: int,
) -> Dict[str, object]:
    x = []
    y = []
    groups = []

    feature_keys = [str(k) for k in feature_lengths]
    target_key = str(target_length)

    for r in rows:
        gid = int(r["starting_point_id"])
        for tn in r["continuation_tempnorms"]:
            x.append([np.log(max(float(tn[k]), 1e-300)) for k in feature_keys])
            y.append(np.log(max(float(tn[target_key]), 1e-300)))
            groups.append(gid)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(x, y, groups=groups))

    reg = LinearRegression()
    reg.fit(x[train_idx], y[train_idx])

    yhat_train = reg.predict(x[train_idx])
    yhat_test = reg.predict(x[test_idx])

    corr = {}
    for j, k in enumerate(feature_keys):
        corr[k] = float(np.corrcoef(x[:, j], y)[0, 1])

    coef_dict = {}
    for j, k in enumerate(feature_keys):
        coef_dict[f"log_t{k}"] = float(reg.coef_[j])

    return {
        "target_length": target_length,
        "feature_lengths": feature_lengths,
        "train_r2_log": float(r2_score(y[train_idx], yhat_train)),
        "test_r2_log": float(r2_score(y[test_idx], yhat_test)),
        "feature_target_correlations_log": corr,
        "coefficients": coef_dict,
        "intercept": float(reg.intercept_),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze early tempnorm predictiveness")
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--feature-lengths", type=int, nargs="+", default=[1, 2, 5, 10])
    p.add_argument("--target-length", type=int, default=20)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if max(args.feature_lengths) >= args.target_length:
        raise ValueError("feature lengths must be strictly smaller than target length")

    rows = read_jsonl(args.data_file)
    q = predictive_analysis(
        rows=rows,
        feature_lengths=args.feature_lengths,
        target_length=args.target_length,
        test_size=args.test_size,
        seed=args.seed,
    )

    report = {
        "predictive_analysis": q,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_lines = [
        "Sanity-check analysis summary",
        f"Predict tempnorm@{args.target_length} from {args.feature_lengths}",
        f"Test R^2 (log-space): {q['test_r2_log']:.6f}",
    ]
    (args.output_dir / "analysis_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
