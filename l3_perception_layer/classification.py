from typing import Sequence
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from ati_config import L3MDEConfig, L3ClassificationConfig
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from urllib.request import urlretrieve

class L3PLayerClassificiation:
    def __init__(
        self,
        l3_config: L3ClassificationConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.l3_config = l3_config
        self.model_name = l3_config.model_name
        self.device = device
        self.model = timm.create_model(self.model_name, pretrained=True).to(self.device)
        self.model.eval()
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
        urlretrieve(url, filename)
        with open("imagenet_classes.txt", "r") as f:
            self.imagenet_classes = [line.strip() for line in f.readlines()]
        self.imagenetlabel_to_isaacsim_label = {
            "beverage": ["water bottle", "beer bottle"],
            "computer": ["laptop", "laptop computer"],
            "banana": ["banana"],
        }
        
    @torch.inference_mode()
    def predict_image(
        self,
        image: np.ndarray,
        gt_bbox:np.ndarray
    ) -> tuple[float, float, str]:
        tensor = self.transform(Image.fromarray(image)).unsqueeze(0).to(self.device) # transform and add batch dimension
        out = self.model(tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        pred_label = self.imagenet_classes[probabilities.cpu().numpy().argmax()]
        return self.__search_object_in_image(
            predicted_label=pred_label,
            gt_bbox=gt_bbox
        ), probabilities.cpu().numpy().max(), pred_label
    
    def __search_object_in_image(
        self, 
        predicted_label: str,
        gt_bbox: np.ndarray,
    ) -> float:
        """Search for the object in the predicted labels within the ground truth bounding box."""
        N = len(gt_bbox['data'])
        idToLabels = gt_bbox['info']['idToLabels']
        scene_label_list = []
        for i in range(N):
            semantic_id, _, _, _, _, _ = gt_bbox['data'][i]
            if not 'class' in idToLabels[str(semantic_id)].keys():
                return -1.0
            _labels = idToLabels[str(semantic_id)]['class'].split(",")
            scene_label_list.extend(_labels)
        count = 0
        for target in self.imagenetlabel_to_isaacsim_label.keys():
            if target in scene_label_list: count += 1
        if count == 0:
            return -1.0
        if predicted_label in scene_label_list:
            return 1.0
        else:   
            return 0.0 
        