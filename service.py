import logging
import os
from os import access

import bentoml
import numpy as np
from PIL import Image

logger = logging.getLogger("bentoml")
logger.setLevel(logging.DEBUG)


@bentoml.service
class YOLOService:
    def __init__(self) -> None:
        self.model = self.load_model("yolo11n.pt")

    def load_model(self, model_path):
        """Load the YOLO model"""
        from ultralytics import YOLO

        return YOLO(model_path)

    @bentoml.api
    async def predict(self, image: Image.Image) -> dict:
        """
        Handle prediction requests

        Args:
            image: PIL Image object

        Returns:
            dict: Prediction results including boxes, scores, and class labels
        """
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Run inference
        results = self.model.predict(img_array)
        result = results[0]  # Get first result since we only process one image

        # Format response
        boxes = []
        for box in result.boxes:
            boxes.append(
                {
                    "xyxy": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                }
            )

        return {"boxes": boxes, "inference_time": float(result.speed["inference"])}
