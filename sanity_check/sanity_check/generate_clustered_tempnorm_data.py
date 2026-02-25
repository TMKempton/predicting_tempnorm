import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .utils import (
    compute_tempnorm,
    final_hidden_from_ids,
    l2_normalize_rows,
    load_model_and_tokenizer,
    load_pickle,
    pure_sample_continuation,
    sample_tokenized_prefixes,
    set_seed,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate clustered tempnorm experiment data")
    p.add_argument("--artifacts-dir", type=Path, required=True)
    p.add_argument("--output-file", type=Path, required=True)
    p.add_argument("--model", type=str, default="facebook/opt-125m")
    p.add_argument("--dataset-name", type=str, default="vblagoje/cc_news")
    p.add_argument("--dataset-config", type=str, default="")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--sample-count", type=int, default=5000)
    p.add_argument("--prefix-len", type=int, default=30)
    p.add_argument("--wander-len", type=int, default=20)
    p.add_argument("--gen-number", type=int, default=8)
    p.add_argument("--gen-length", type=int, default=20)
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--eval-lengths", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if max(args.eval_lengths) > args.gen_length:
        raise ValueError(
            f"max(eval_lengths)={max(args.eval_lengths)} exceeds gen_length={args.gen_length}"
        )

    pca = load_pickle(args.artifacts_dir / "pca.pkl")
    kmeans = load_pickle(args.artifacts_dir / "kmeans.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    prefixes = sample_tokenized_prefixes(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        n_samples=args.sample_count,
        prefix_len=args.prefix_len,
        seed=args.seed,
    )

    rows: List[Dict] = []
    for i, prefix in enumerate(tqdm(prefixes, desc="Generating data")):
        wander_ids, _ = pure_sample_continuation(model, prefix, args.wander_len, device)
        starting_ids = prefix + wander_ids

        hidden = final_hidden_from_ids(model, starting_ids, device)
        hidden_norm = l2_normalize_rows(np.expand_dims(hidden, axis=0))[0]
        pc_activation = pca.transform(np.expand_dims(hidden_norm, axis=0))[0]
        cluster = int(kmeans.predict(np.expand_dims(pc_activation, axis=0))[0])

        continuation_tempnorms: List[Dict[str, float]] = []
        for _ in range(args.gen_number):
            _, probs = pure_sample_continuation(model, starting_ids, args.gen_length, device)
            tn = {str(L): compute_tempnorm(probs, eval_length=L, alpha=args.alpha) for L in args.eval_lengths}
            continuation_tempnorms.append(tn)

        row = {
            "starting_point_id": i,
            "prefix_token_ids": prefix,
            "starting_token_ids": starting_ids,
            "pc_activation": [float(x) for x in pc_activation.tolist()],
            "cluster": cluster,
            "continuation_tempnorms": continuation_tempnorms,
            "config": {
                "model": args.model,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_split": args.dataset_split,
                "prefix_len": args.prefix_len,
                "wander_len": args.wander_len,
                "gen_number": args.gen_number,
                "gen_length": args.gen_length,
                "alpha": args.alpha,
                "eval_lengths": args.eval_lengths,
            },
        }
        rows.append(row)

    write_jsonl(args.output_file, rows)


if __name__ == "__main__":
    main()
