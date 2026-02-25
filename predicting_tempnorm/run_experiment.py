import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .config import ExperimentConfig, normalize_model_name


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pure_sample_continuation(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    num_new_tokens: int,
    device: torch.device,
) -> Tuple[List[int], List[float]]:
    generated: List[int] = []
    probs_out: List[float] = []

    with torch.no_grad():
        out = model(input_ids=prompt_ids.to(device), use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

        for _ in range(num_new_tokens):
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_prob = probs.gather(dim=-1, index=next_token)

            token_id = int(next_token.item())
            prob_value = float(next_prob.item())
            generated.append(token_id)
            probs_out.append(prob_value)

            out = model(input_ids=next_token, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

    return generated, probs_out


def get_last_hidden_vector(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device: torch.device,
) -> List[float]:
    with torch.no_grad():
        out = model(input_ids=input_ids.to(device), output_hidden_states=True, use_cache=False)
        vec = out.hidden_states[-1][0, -1, :].to(dtype=torch.float16).cpu()
    return vec.tolist()


def compute_tempnorm_summary(generations: List[Dict], eval_lengths: List[int], alpha: float) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    power = alpha - 1.0

    for eval_length in eval_lengths:
        individual_tempnorms: List[float] = []
        for gen in generations:
            probs = gen["model_probs"]
            if eval_length > len(probs):
                raise ValueError(
                    f"eval_length={eval_length} exceeds available probabilities={len(probs)}"
                )

            # Compute product_{t<=L} p_t^(alpha-1) in log-space for numerical stability.
            log_tempnorm = power * sum(math.log(max(float(p), 1e-45)) for p in probs[:eval_length])
            individual_tempnorms.append(float(math.exp(log_tempnorm)))

        summary[f"individual_tempnorms_{eval_length}"] = individual_tempnorms
        summary[f"tempnorm_mean_{eval_length}"] = float(np.mean(individual_tempnorms))
        summary[f"tempnorm_variance_{eval_length}"] = float(np.var(individual_tempnorms))

    return summary


def read_contexts(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    contexts = [line.strip() for line in lines if line.strip()]
    if not contexts:
        raise ValueError(f"No non-empty contexts found in {path}")
    return contexts


def run_experiment(config: ExperimentConfig, contexts: List[str], output_file: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = normalize_model_name(config.model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for idx, context in enumerate(tqdm(contexts, desc="Contexts")):
            context_ids = tokenizer(context, return_tensors="pt").input_ids

            wander_ids, _ = pure_sample_continuation(
                model=model,
                prompt_ids=context_ids,
                num_new_tokens=config.initial_wander,
                device=device,
            )

            starting_point_ids = torch.cat(
                [context_ids, torch.tensor([wander_ids], dtype=torch.long)],
                dim=1,
            )
            starting_point = tokenizer.decode(starting_point_ids[0], skip_special_tokens=True)
            starting_hidden_vector = get_last_hidden_vector(model, starting_point_ids, device)

            generations: List[Dict] = []
            for gen_i in range(config.gen_number):
                gen_ids, model_probs = pure_sample_continuation(
                    model=model,
                    prompt_ids=starting_point_ids,
                    num_new_tokens=config.gen_length,
                    device=device,
                )
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                generations.append(
                    {
                        "generation_index": gen_i,
                        "token_ids": gen_ids,
                        "generated_text": generated_text,
                        "model_probs": model_probs,
                    }
                )

            tempnorm_summary = compute_tempnorm_summary(
                generations=generations,
                eval_lengths=config.eval_lengths,
                alpha=config.alpha,
            )

            record = {
                "context_index": idx,
                "initial_context": context,
                "starting_point": starting_point,
                "starting_hidden_vector_dtype": "float16",
                "starting_hidden_vector": starting_hidden_vector,
                "generations": generations,
                "config": asdict(config),
            }
            record.update(tempnorm_summary)
            f.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Predicting TempNorm data-generation experiment")
    p.add_argument("--contexts-file", type=Path, required=True)
    p.add_argument("--output-file", type=Path, required=True)
    p.add_argument("--model", type=str, default="facebook/opt-125m")
    p.add_argument("--gen-length", type=int, default=100)
    p.add_argument("--eval-lengths", type=int, nargs="+", default=[1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100])
    p.add_argument("--gen-number", type=int, default=8)
    p.add_argument("--alpha", type=int, default=4)
    p.add_argument("--initial-wander", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = ExperimentConfig(
        model=args.model,
        gen_length=args.gen_length,
        eval_lengths=args.eval_lengths,
        gen_number=args.gen_number,
        alpha=args.alpha,
        initial_wander=args.initial_wander,
        seed=args.seed,
    )

    contexts = read_contexts(args.contexts_file)
    run_experiment(config=config, contexts=contexts, output_file=args.output_file)


if __name__ == "__main__":
    main()