import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class DateDetector:
    """
    A wrapper class for the YOLOv8 object detection model specifically 
    tailored for Expiry Date detection.
    """
    def __init__(self, weights_path: str):
        """
        Initializes the YOLO model. If weights_path is a standard YOLO model 
        (like yolov8n.pt), it downloads it automatically. If it's a local path, 
        it loads the local weights.
        """
        logger.info(f"Initializing DateDetector with weights: {weights_path}")
        self.model = YOLO(weights_path)

    def train(self, **kwargs):
        """
        Starts the training process. 
        Passes all keyword arguments directly to Ultralytics YOLO engine.
        """
        logger.info("Starting YOLO training process...")
        results = self.model.train(**kwargs)
        return results

    def predict(self, source, **kwargs):
        """
        Runs inference on the provided source (image path, PIL image, cv2 frame).
        """
        return self.model.predict(source, **kwargs)

    def export(self, format="onnx"):
        """
        Exports the model to a different format (e.g., ONNX, TensorRT).
        """
        logger.info(f"Exporting model to {format} format...")
        return self.model.export(format=format)