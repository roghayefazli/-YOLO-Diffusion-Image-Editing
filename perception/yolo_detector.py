# image: np.ndarray or PIL.Image
# {
#   "image_size": (H, W),
#   "objects": [
#     {
#       "class": "person",
#       "class_id": 0,
#       "bbox": [x1, y1, x2, y2],
#       "confidence": 0.93,
#       "mask": None,          # future: segmentation
#       "keypoints": None      # future: pose
#     }
#   ]
# }

# Step 0: Project structure skeleton
# We will build this step by step


# ==============================
# Step 2: YOLO Perception Module
# File: perception/yolo_detector.py
# Detection only (bbox + class)
# ==============================
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image


class YOLODetector:
    def __init__(self, model_name: str = "yolov8n-seg.pt", device: str = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model = YOLO(model_name)
        self.model.to(self.device)

    def extract_main_person(self, image):
        """
        Returns:
            person_rgba (PIL Image with transparent background)
            mask (PIL Image)
        """

        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        results = self.model(image_np, verbose=False)[0]

        if results.masks is None:
            raise ValueError("No segmentation masks found.")

        H, W = image_np.shape[:2]

        best_person = None
        best_conf = 0
        best_mask = None

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name == "person" and conf > best_conf:
                best_conf = conf
                best_person = box
                best_mask = results.masks.data[i].cpu().numpy()

        if best_mask is None:
            raise ValueError("No person detected.")

        # تبدیل mask به تصویر
        mask = (best_mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask).resize((W, H))

        # ساخت RGBA با پس‌زمینه شفاف
        person_rgba = image.convert("RGBA")
        person_rgba.putalpha(mask_img)

        return person_rgba, mask_img