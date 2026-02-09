"""
Computes accuracy and token usage from doctor-patient dialogues.
"""

from typing import Dict

from transformers import AutoTokenizer


# Use the same tokenizer as the Llama 3.1 expert/patient model
_TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


def get_tokens_count(dialogue, tokenizer):
    doctor_tokens_count = 0
    for name, replic in dialogue:
        if name == "doctor" and replic is not None:
            text = replic if isinstance(replic, str) else ""
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
    
    correct_answs = 0
    dialogue_length = 0
    for dialogue_i, diagnosis_i, ground_truth_i in zip(dialogues, diagnoses, ground_truth):
        if diagnosis_i == ground_truth_i:
            correct_answs += 1
        dialogue_length += get_tokens_count(dialogue_i, _TOKENIZER)
    
    return {
        "fitness": correct_answs / len(ground_truth),
        "tokens_count": dialogue_length / len(ground_truth),
        "is_valid": 1
    }
