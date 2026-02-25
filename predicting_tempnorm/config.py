from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    model: str = "facebook/opt-125m"
    gen_length: int = 100
    eval_lengths: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100])
    gen_number: int = 8
    alpha: int = 4
    initial_wander: int = 30
    seed: int = 42


def normalize_model_name(model_name: str) -> str:
    alias = model_name.strip().lower()
    if alias in {"opt-125m", "facebook/opt-125m"}:
        return "facebook/opt-125m"
    return model_name
