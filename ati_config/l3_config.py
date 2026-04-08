from dataclasses import dataclass

@dataclass(frozen=True)
class L3MDEConfig:
    reward_type: str = "flipped"  # Options: ["flipped", "test_time_augment", "oracle"]
    max_depth: float = 80.0
    min_depth: float = 0.01
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"  # Options: "depth-anything/Depth-Anything-V2-Small-hf", "depth-anything/Depth-Anything-V2-Base-hf"
    shift_ratios: tuple[float, ...] = (-0.1, 0.1)
    zoom_factors: tuple[float, ...] = (0.8, 1.2)
    gaussian_noise_stds: tuple[float, ...] = (0.05, 0.1)
    brightness_factors: tuple[float, ...] = (0.75,)
    color_jitter_strengths: tuple[float, ...] = (0.25,)
    disable_hflip: bool = False 
    prediction_mode: str = "mean"  # Options: "mean", "identity"
    
    def __post_init__(self):
        if self.reward_type not in ["flipped", "test_time_augment", "oracle"]:
            raise ValueError(f"Invalid reward_type: {self.reward_type}. Must be one of ['flipped', 'test_time_augment', 'oracle']")
        if self.model_name not in [
            "depth-anything/Depth-Anything-V2-Small-hf", 
            "depth-anything/Depth-Anything-V2-Base-hf"
        ]:
            raise ValueError(f"Invalid model_name: {self.model_name}. Must be one of ['depth-anything/Depth-Anything-V2-Small-hf', 'depth-anything/Depth-Anything-V2-Base-hf']")