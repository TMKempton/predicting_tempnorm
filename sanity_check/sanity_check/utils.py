import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer


def text_stream(dataset_name: str, dataset_config: str, split: str = "train") -> Iterable[str]:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    for row in ds:
        txt = row.get("text", "")
        if isinstance(txt, str) and txt.strip():
            yield txt


def sample_tokenized_prefixes(
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    n_samples: int,
    prefix_len: int,
    seed: int,
) -> List[List[int]]:
    set_seed(seed)
    prefixes: List[List[int]] = []

    for text in text_stream(dataset_name, dataset_config, split="train"):
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) < prefix_len:
            continue
        prefixes.append(token_ids[:prefix_len])
        if len(prefixes) >= n_samples:
            break

    if len(prefixes) < n_samples:
        raise ValueError(f"Only found {len(prefixes)} valid texts; need {n_samples}.")
    return prefixes


def final_hidden_from_ids(model, input_ids_1d: List[int], device: torch.device) -> np.ndarray:
    x = torch.tensor([input_ids_1d], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=x, output_hidden_states=True, use_cache=False)
        vec = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
    return vec


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def pure_sample_continuation(model, prompt_ids_1d: List[int], num_new_tokens: int, device: torch.device) -> Tuple[List[int], List[float]]:
    prompt = torch.tensor([prompt_ids_1d], dtype=torch.long, device=device)
    generated: List[int] = []
    probs_out: List[float] = []

    with torch.no_grad():
        out = model(input_ids=prompt, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        for _ in range(num_new_tokens):
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_prob = probs.gather(dim=-1, index=next_token)

            generated.append(int(next_token.item()))
            probs_out.append(float(next_prob.item()))

            out = model(input_ids=next_token, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

    return generated, probs_out


def compute_tempnorm(model_probs: List[float], eval_length: int, alpha: float) -> float:
    power = alpha - 1.0
    sub = model_probs[:eval_length]
    if len(sub) < eval_length:
        raise ValueError(f"Need {eval_length} probs, got {len(sub)}")
    log_val = power * sum(math.log(max(float(p), 1e-45)) for p in sub)
    return float(math.exp(log_val))


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)