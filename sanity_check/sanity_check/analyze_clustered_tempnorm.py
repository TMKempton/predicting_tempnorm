import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

from .utils import read_jsonl


def flatten_tempnorm50(rows: List[Dict]) -> np.ndarray:
    vals = []
    for r in rows:
        for tn in r["continuation_tempnorms"]:
            vals.append(float(tn["50"]))
    return np.asarray(vals, dtype=np.float64)


def per_starting_point_stats(rows: List[Dict]) -> Dict[str, np.ndarray]:
    means = []
    vars_ = []
    clusters = []
    for r in rows:
        vals = np.asarray([float(tn["50"]) for tn in r["continuation_tempnorms"]], dtype=np.float64)
        means.append(float(np.mean(vals)))
        vars_.append(float(np.var(vals)))
        clusters.append(int(r["cluster"]))
    return {
        "mean50": np.asarray(means, dtype=np.float64),
        "var50_within_starting_point": np.asarray(vars_, dtype=np.float64),
        "cluster": np.asarray(clusters, dtype=np.int64),
    }


def cluster_variance_decomposition(point_means: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
    global_var = float(np.var(point_means))
    uniq = np.unique(clusters)

    weighted_within = 0.0
    cluster_stats = {}
    n_total = len(point_means)

    for c in uniq:
        mask = clusters == c
        vals = point_means[mask]
        c_var = float(np.var(vals)) if len(vals) > 1 else 0.0
        c_mean = float(np.mean(vals)) if len(vals) > 0 else 0.0
        cluster_stats[str(int(c))] = {
            "count": int(np.sum(mask)),
            "mean": c_mean,
            "variance": c_var,
        }
        weighted_within += (len(vals) / n_total) * c_var

    reduction = 0.0
    if global_var > 0:
        reduction = 1.0 - (weighted_within / global_var)

    return {
        "global_variance_point_means": global_var,
        "weighted_within_cluster_variance": float(weighted_within),
        "variance_reduction_fraction": float(reduction),
        "cluster_stats": cluster_stats,
    }


def compare_cluster_vs_within_point(rows: List[Dict], point_means: np.ndarray, point_vars: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
    uniq = np.unique(clusters)
    cluster_vars = []
    for c in uniq:
        vals = point_means[clusters == c]
        cluster_vars.append(float(np.var(vals)) if len(vals) > 1 else 0.0)

    avg_cluster_var = float(np.mean(cluster_vars)) if cluster_vars else 0.0
    avg_within_point_var = float(np.mean(point_vars)) if len(point_vars) else 0.0

    ratio = None
    if avg_within_point_var > 0:
        ratio = float(avg_cluster_var / avg_within_point_var)

    return {
        "avg_cluster_variance_of_point_means": avg_cluster_var,
        "avg_within_starting_point_variance_over_8_continuations": avg_within_point_var,
        "cluster_to_within_point_variance_ratio": ratio,
    }


def predictive_analysis(rows: List[Dict], test_size: float, seed: int) -> Dict[str, float]:
    x = []
    y = []
    groups = []

    for r in rows:
        gid = int(r["starting_point_id"])
        for tn in r["continuation_tempnorms"]:
            x.append([
                np.log(max(float(tn["1"]), 1e-300)),
                np.log(max(float(tn["2"]), 1e-300)),
                np.log(max(float(tn["5"]), 1e-300)),
                np.log(max(float(tn["10"]), 1e-300)),
            ])
            y.append(np.log(max(float(tn["50"]), 1e-300)))
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
    for j, k in enumerate(["1", "2", "5", "10"]):
        corr[k] = float(np.corrcoef(x[:, j], y)[0, 1])

    return {
        "train_r2_log": float(r2_score(y[train_idx], yhat_train)),
        "test_r2_log": float(r2_score(y[test_idx], yhat_test)),
        "feature_target_correlations_log": corr,
        "coefficients": {
            "log_t1": float(reg.coef_[0]),
            "log_t2": float(reg.coef_[1]),
            "log_t5": float(reg.coef_[2]),
            "log_t10": float(reg.coef_[3]),
        },
        "intercept": float(reg.intercept_),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze clustered tempnorm data")
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data_file)

    all_tempnorm50 = flatten_tempnorm50(rows)
    point = per_starting_point_stats(rows)

    q1 = {
        "global_tempnorm50_mean_over_all_continuations": float(np.mean(all_tempnorm50)),
        "global_tempnorm50_variance_over_all_continuations": float(np.var(all_tempnorm50)),
    }
    q1.update(cluster_variance_decomposition(point["mean50"], point["cluster"]))

    q2 = compare_cluster_vs_within_point(rows, point["mean50"], point["var50_within_starting_point"], point["cluster"])
    q3 = predictive_analysis(rows, test_size=args.test_size, seed=args.seed)

    report = {
        "question_1_cluster_partitioning": q1,
        "question_2_cluster_vs_within_starting_point": q2,
        "question_3_predict_tempnorm50_from_early_tempnorms": q3,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_lines = [
        "Sanity-check analysis summary",
        f"Q1 variance reduction fraction: {q1['variance_reduction_fraction']:.6f}",
        f"Q2 cluster/within-point variance ratio: {q2['cluster_to_within_point_variance_ratio']}",
        f"Q3 test R^2 (log-space): {q3['test_r2_log']:.6f}",
    ]
    (args.output_dir / "analysis_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()