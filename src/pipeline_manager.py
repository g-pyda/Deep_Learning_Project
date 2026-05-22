import cv2
import yaml
import logging
import argparse
import os
from pathlib import Path

# Import our custom modules
from models.date_detector import DateDetector
from pipeline_ocr import ExpiryOCR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineManager:
    """
    End-to-end pipeline orchestrator for Expiry Date Detection and Recognition.
    """
    def __init__(self, detector_config_path: str, ocr_config_path: str):
        logger.info("Initializing Pipeline Manager...")
        
        # Load Detector
        det_config = self._load_config(detector_config_path)
        weights_path = det_config.get("model", {}).get("weights", "yolov8n.pt")
        
        # Ensure we are using the best.pt from the training phase if specified
        if not os.path.exists(weights_path) and "best.pt" in weights_path:
            logger.error(f"Detector weights not found at {weights_path}. Train the model first.")
            raise FileNotFoundError(f"Missing weights: {weights_path}")
            
        self.detector = DateDetector(weights_path=weights_path)
        
        # Load OCR
        self.ocr_engine = ExpiryOCR(config_path=ocr_config_path)
        logger.info("Pipeline initialized successfully.")

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def run_pipeline(self, image_path: str) -> dict:
        """
        Executes the full pipeline on a single image.
        Returns a dictionary containing the detected text, confidence, and bounding box coordinates.
        """
        logger.info(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # 1. Detect Bounding Box
        results = self.detector.predict(image, verbose=False)
        
        # Check if any objects were detected
        if len(results) == 0 or len(results[0].boxes) == 0:
            logger.warning("No expiry date found on the image.")
            return {"text": None, "bbox": None, "confidence": 0.0}

        # Take the detection with the highest confidence (usually the first one)
        best_box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
        confidence = float(best_box.conf[0].cpu().numpy())

        logger.debug(f"Detected bounding box at [{x1}, {y1}, {x2}, {y2}] with conf: {confidence:.2f}")

        # 2. Crop Region of Interest (ROI)
        # Ensure coordinates are within image boundaries
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            logger.warning("Cropped bounding box is empty.")
            return {"text": None, "bbox": [x1, y1, x2, y2], "confidence": confidence}

        # 3. Process Crop & OCR
        recognized_text = self.ocr_engine.process(crop)
        logger.info(f"Extracted Text: {recognized_text}")

        return {
            "text": recognized_text,
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end Expiry Date Pipeline.")
    parser.add_argument("--img", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--det_config", type=str, default="config/detector_config.yaml", help="Path to detector config.")
    parser.add_argument("--ocr_config", type=str, default="config/ocr_config.yaml", help="Path to OCR config.")
    args = parser.parse_args()
    
    pipeline = PipelineManager(args.det_config, args.ocr_config)
    result = pipeline.run_pipeline(args.img)
    
    print("\n--- Final Pipeline Result ---")
    print(f"Text:       {result['text']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"BBox:       {result['bbox']}")