from typing import Sequence
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from ati_config import L3MDEConfig
from l3_perception_layer.mde_utils.test_time_augment import (
    TTATransform,
    default_recommended_tta_transforms,
    apply_tta_transform,
    invert_tta_depth_transform,
    select_eval_prediction
)

class L3PLayerDepthAnythingv2:
    def __init__(
        self,
        l3_config: L3MDEConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.l3_config = l3_config
        self.model_name = l3_config.model_name
        self.device = device
        self.min_depth = l3_config.min_depth
        self.max_depth = l3_config.max_depth
        self.reward_type = l3_config.reward_type
        if not self.reward_type in ["flipped", "test_time_augment"]:
            raise ValueError(f"Invalid reward_type: {self.reward_type}. Must be one of ['flipped', 'test_time_augment']")
        if not self.model_name in [
            "depth-anything/Depth-Anything-V2-Small-hf", 
            "depth-anything/Depth-Anything-V2-Base-hf"
        ]:
            raise ValueError(f"Invalid model_name: {self.model_name}. Must be one of ['depth-anything/Depth-Anything-V2-Small-hf', 'depth-anything/Depth-Anything-V2-Base-hf']")
        self.model = pipeline("depth-estimation", model=self.model_name, device=0 if self.device == "cuda" else -1)
        self.transforms = self.__build_recommended_transform_list()
    
    
    def __build_recommended_transform_list(self) -> list[TTATransform]:
        return default_recommended_tta_transforms(
            include_identity=True,
            include_hflip=not self.l3_config.disable_hflip,
            shift_ratios=tuple(self.l3_config.shift_ratios),
            zoom_factors=tuple(self.l3_config.zoom_factors),
            gaussian_noise_stds=tuple(self.l3_config.gaussian_noise_stds),
            brightness_factors=tuple(self.l3_config.brightness_factors),
            color_jitter_strengths=tuple(self.l3_config.color_jitter_strengths),
        )

    def predict_depth(
        self, 
        images: Sequence[Image.Image | np.ndarray],
        gt_depth: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        if self.reward_type == "flipped":
            pred_depths = self.__predict_depth_flipped(images)
            eval_pred_depth = pred_depths[0]
        elif self.reward_type == "test_time_augment":
            pred_depths = self.__predict_depth_with_tta(images)
            eval_pred_depth = select_eval_prediction(
                inverse_depths=pred_depths[0],
                transforms=self.transforms,
                prediction_mode=self.l3_config.prediction_mode,
            )
        else:
            raise ValueError(f"Invalid reward_type: {self.reward_type}. Must be one of ['flipped', 'test_time_augment']")
        
        metric_info = self.__evaluate_l3_depth(
            eval_pred_depth,
            gt_depth,
            save_path=None,
            save_files=False,
            verbose=True,
        )
        return pred_depths, metric_info
        

    def __predict_depth_flipped(self, images: Sequence[Image.Image | np.ndarray]) -> list[np.ndarray]:
        rgb = [ images[0], images[0].transpose(Image.FLIP_LEFT_RIGHT) ]
        predictions = self.model(rgb)
        return [
            np.array(predictions[0]["depth"]),
            np.array(predictions[1]["depth"])
        ]

    def __predict_depth_with_tta(self, images: Sequence[Image.Image | np.ndarray]) -> list[np.ndarray]:
        infer_list = self.__build_tta_inference_batch(images)
        predictions = self.model(infer_list)
        inverse_depths = self.__invert_tta_depth_predictions(predictions, num_original_images=len(images))
        return inverse_depths

    def __build_tta_inference_batch(
        self,
        images: Sequence[Image.Image | np.ndarray],
    ) -> tuple[list[Image.Image], list[TTATransform]]:
        transforms = list(self.transforms)
        infer_list: list[Image.Image] = []

        for image in images:
            for transform in transforms:
                infer_list.append(apply_tta_transform(image, transform))
        return infer_list

    def __invert_tta_depth_predictions(
        self,
        predictions: Sequence[dict | np.ndarray],
        num_original_images: int,
    ) -> list[list[np.ndarray]]:
        transforms = list(self.transforms)
        expected = num_original_images * len(transforms)
        if len(predictions) != expected:
            raise ValueError(
                f"Expected {expected} predictions for {num_original_images} images and "
                f"{len(transforms)} transforms, got {len(predictions)}"
            )

        inverse_depths: list[list[np.ndarray]] = []
        for image_idx in range(num_original_images):
            start = image_idx * len(transforms)
            grouped: list[np.ndarray] = []
            for offset, transform in enumerate(transforms):
                pred = predictions[start + offset]
                depth = np.asarray(pred["depth"] if isinstance(pred, dict) else pred)
                grouped.append(invert_tta_depth_transform(depth, transform))
            inverse_depths.append(grouped)
        return inverse_depths

    def __compute_errors_numpy(
        self,
        gt,
        pred,
        align_mode="scale_shift",
        eps=1e-8,
    ):
        gt = np.asarray(gt).astype(np.float64)
        pred = np.asarray(pred).astype(np.float64)

        valid = np.isfinite(gt) & np.isfinite(pred)
        valid &= (gt > self.min_depth) & (gt < self.max_depth)
        valid &= (pred > 0)
        if valid.sum() == 0:
            raise ValueError("No valid pixels found for evaluation.")

        gt_valid = gt[valid]
        pred_valid = pred[valid]

        if align_mode == "scale_shift":
            a = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
            x, _, _, _ = np.linalg.lstsq(a, gt_valid, rcond=None)
            scale, shift = x
            pred_valid = scale * pred_valid + shift
        elif align_mode is not None:
            raise ValueError(f"Unknown align_mode: {align_mode}")

        pred_valid = np.clip(pred_valid, self.min_depth, self.max_depth)
        gt_valid = np.clip(gt_valid, self.min_depth, self.max_depth)

        thresh = np.maximum(gt_valid / (pred_valid + eps), pred_valid / (gt_valid + eps))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()
        rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
        rmse_log = np.sqrt(np.mean((np.log(gt_valid + eps) - np.log(pred_valid + eps)) ** 2))
        abs_rel = np.mean(np.abs(gt_valid - pred_valid) / (gt_valid + eps))
        sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / (gt_valid + eps))

        return {
            "abs_rel": float(abs_rel),
            "sq_rel": float(sq_rel),
            "rmse": float(rmse),
            "rmse_log": float(rmse_log),
            "a1": float(a1),
            "a2": float(a2),
            "a3": float(a3),
        }


    def __evaluate_l3_depth(
        self, 
        pred_depth, 
        gt_depth, 
        save_path="", 
        steps=0, 
        save_files=False, 
        verbose=False
    ):
        if save_files:
            np.save(f"{save_path}/pred_mde_{steps:03d}.npy", pred_depth)

        pred_depth = pred_depth.flatten()
        gt_depth = gt_depth.flatten()

        mask = np.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth  )
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        gt_depth = 1 / gt_depth

        metrics = self.__compute_errors_numpy(gt_depth, pred_depth)
        if verbose:
            print(
                "[MDE Result on step {:03d}] | abs_rel: {:.2f} | sq_rel {:.2f} | rmse {:.2f} | "
                "rmse_log {:.2f} | a1 {:.2f} | a2 {:.2f} | a3 {:.2f} |".format(
                    steps,
                    metrics["abs_rel"],
                    metrics["sq_rel"],
                    metrics["rmse"],
                    metrics["rmse_log"],
                    metrics["a1"],
                    metrics["a2"],
                    metrics["a3"],
                )
            )
        return metrics