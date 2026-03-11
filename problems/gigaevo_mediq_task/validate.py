"""
Computes accuracy and token usage from doctor-patient dialogues.
"""

from typing import Dict
import os
from transformers import AutoTokenizer

# Lazy-load so multiple processes (e.g. run_all_experts.sh) don't block each other on import.
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        model_name = os.environ.get("MEDIQ_MODEL_NAME")
        if model_name and model_name.lower().startswith("meta-llama/"):
            _, suffix = model_name.split("/", 1)
            model_name = f"meta-llama/{suffix}"
        # Use HF_TOKEN for gated models (e.g. Meta-Llama); set in run.sh or env.
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        _TOKENIZER = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )
    return _TOKENIZER


def get_tokens_count(dialogue, tokenizer):
    doctor_tokens_count = 0
    for name, replic in dialogue:
        if name == "doctor" and replic is not None:
            try:
                text = replic if isinstance(replic, str) else str(replic)
            except Exception:
                text = ""
            replic_tokens = tokenizer.encode(text, add_special_tokens=False)
            doctor_tokens_count += len(replic_tokens)
    return doctor_tokens_count


def validate(data) -> Dict[str, float]:
    """
    Compute accuracy and mean tokens count from the dialogues.

    Args:
        data: Tuple of (dialogues, diagnoses, case_ids, ground_truth) returned by entrypoint().
    """
    # Unpack all values from entrypoint()
    dialogues, diagnoses, _case_ids, ground_truth = data
    
    if len(ground_truth) != len(diagnoses):
        raise ValueError(f"Number of cases for predictions and ground truth must match! Got {len(diagnoses)} for predictions, should be {len(ground_truth)}")
    
    tokenizer = _get_tokenizer()
    correct_answs = 0
    dialogue_length = 0
    for dialogue_i, diagnosis_i, ground_truth_i in zip(dialogues, diagnoses, ground_truth):
        if diagnosis_i == ground_truth_i:
            correct_answs += 1
        dialogue_length += get_tokens_count(dialogue_i, tokenizer)
    
    return {
        "fitness": correct_answs / len(ground_truth),
        "tokens_count": dialogue_length / len(ground_truth),
        "is_valid": 1
    }
