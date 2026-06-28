
# ==============================
# Step 3: Conditioning Encoder
# File: conditioning/condition_encoder.py
# Converts YOLO detections to diffusion conditions
# ==============================

import torch
import numpy as np

class ConditionEncoder:
    def __init__(self, image_size, device="cpu"):
        self.H, self.W = image_size
        self.device = device

    def build_layout_map(self, objects):
        """Binary layout map from bounding boxes"""
        layout = torch.zeros((1, 1, self.H, self.W), device=self.device)
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            layout[:, :, y1:y2, x1:x2] = 1.0
        return layout

    def build_semantic_tokens(self, objects):
        """Simple semantic encoding using class IDs"""
        class_ids = [obj["class_id"] for obj in objects]
        return torch.tensor(class_ids, device=self.device)

    def encode(self, scene_dict):
        objects = scene_dict["objects"]
        layout_map = self.build_layout_map(objects)
        semantic_tokens = self.build_semantic_tokens(objects)
        return {
            "layout_map": layout_map,
            "semantic_tokens": semantic_tokens
        }
