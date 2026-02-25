import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from .utils import (
    final_hidden_from_ids,
    l2_normalize_rows,
    load_model_and_tokenizer,
    sample_tokenized_prefixes,
    save_pickle,
    set_seed,
)
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PCA/KMeans reference space from hidden states")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="facebook/opt-125m")
    p.add_argument("--dataset-name", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--sample-count", type=int, default=2000)
    p.add_argument("--token-index", type=int, default=50, help="0-based index; 50 means 51st token")
    p.add_argument("--pca-components", type=int, default=30)
    p.add_argument("--num-clusters", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    prefix_len = args.token_index + 1
    prefixes = sample_tokenized_prefixes(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        n_samples=args.sample_count,
        prefix_len=prefix_len,
        seed=args.seed,
    )

    activations = []
    for ids in tqdm(prefixes, desc="Extracting activations"):
        activations.append(final_hidden_from_ids(model, ids, device))

    x = np.stack(activations, axis=0)
    x_norm = l2_normalize_rows(x)

    pca = PCA(n_components=args.pca_components, random_state=args.seed)
    pc = pca.fit_transform(x_norm)

    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=20)
    labels = kmeans.fit_predict(pc)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.save(args.output_dir / "reference_activations.npy", x_norm.astype(np.float32))
    np.save(args.output_dir / "reference_pc_activations.npy", pc.astype(np.float32))
    np.save(args.output_dir / "reference_cluster_labels.npy", labels.astype(np.int32))

    save_pickle(args.output_dir / "pca.pkl", pca)
    save_pickle(args.output_dir / "kmeans.pkl", kmeans)

    metadata = {
        "model": args.model,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "sample_count": args.sample_count,
        "token_index": args.token_index,
        "pca_components": args.pca_components,
        "num_clusters": args.num_clusters,
        "seed": args.seed,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    (args.output_dir / "reference_metadata.json").write_text(__import__("json").dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()