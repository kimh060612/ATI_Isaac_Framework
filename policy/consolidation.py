from collections import Counter
import json
import os
import numpy as np

def build_reward_override(reward_history: list[dict]) -> dict:
    if not reward_history:
        return {
            "reward": 0.0,
            "image_reward": 0.0,
            "depth_reward": 0.0,
            "uncertainty": 0.0,
        }
    return {
        key: float(np.mean([
            metric[key] for metric in reward_history
        ])) 
        for key in reward_history[0].keys()
    }


def load_heuristic_memory(memory_path: str) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    if not os.path.exists(memory_path):
        return {}, {}
    with open(memory_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    heuristic_offsets = {
        state: (float(offset[0]), float(offset[1]))
        for state, offset in payload.get("heuristic_offsets", {}).items()
        if isinstance(offset, (list, tuple)) and len(offset) == 2
    }
    state_update_counts = {
        state: int(count)
        for state, count in payload.get("state_update_counts", {}).items()
    }
    return heuristic_offsets, state_update_counts


def save_heuristic_memory(
    memory_path: str,
    heuristic_offsets: dict[str, tuple[float, float]],
    state_update_counts: dict[str, int],
) -> None:
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    payload = {
        "heuristic_offsets": {
            state: [float(offset[0]), float(offset[1])]
            for state, offset in sorted(heuristic_offsets.items())
        },
        "state_update_counts": {
            state: int(count)
            for state, count in sorted(state_update_counts.items())
        },
    }
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)