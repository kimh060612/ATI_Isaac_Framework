from typing import Dict, List
import carb.settings
import numpy as np
import os

def __make_dirs(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def configure_isaac_sim_logging():
    carb.settings.get_settings().set("/log/debugConsoleLevel", "Fatal")  # verbose"|"info"|"warning"|"error"|"fatal"
    carb.settings.get_settings().set("/log/enabled", False)
    carb.settings.get_settings().set("/log/outputStreamLevel", "Error")
    carb.settings.get_settings().set("/log/fileLogLevel", "Error")
    carb.settings.get_settings().set("/app/enableDeveloperWarnings", False)
    carb.settings.get_settings().set("/app/scripting/ignoreWarningDialog", True)
    carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/verbose", False)
    carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/info", False)
    carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/warning", False)
    carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/error", False)
    carb.settings.get_settings().set("/exts/omni.kit.window.console/logFilter/fatal", False)

def save_synthetic_data(
    DATA_PATH: str,
    syn_data: Dict[str, List[np.ndarray]],
    lap_idx: int,
) -> None:
    __make_dirs(DATA_PATH)
    for key, array_list in syn_data.items():
        for i, array in enumerate(array_list):
            filename = f"{key}_lap{lap_idx:03d}_{i:03d}.npy"
            save_path = os.path.join(DATA_PATH, key, filename)
            __make_dirs(os.path.dirname(save_path))
            np.save(save_path, array)

def save_compressed_synthetic_data(
    DATA_PATH: str,
    syn_data: Dict[str, List[np.ndarray]],
    lap_idx: int,
) -> None:
    __make_dirs(DATA_PATH)
    for key, array_list in syn_data.items():
        for i, array in enumerate(array_list):
            filename = f"{key}_lap{lap_idx:03d}_{i:03d}.npz"
            save_path = os.path.join(DATA_PATH, key, filename)
            __make_dirs(os.path.dirname(save_path))
            np.savez_compressed(save_path, array=array)

def get_eval_averages(
    metrics_list: dict,
    key_category: str = None,
) -> dict:
    avg_metrics = {}
    for key in metrics_list[0].keys():
        new_key = f"{key_category}/{key}" if key_category else key
        avg_metrics[new_key] = np.mean([metric[key] for metric in metrics_list])
    return avg_metrics
