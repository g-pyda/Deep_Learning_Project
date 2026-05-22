import cv2
import numpy as np
import easyocr
import yaml
import logging
import os

logger = logging.getLogger(__name__)

class ExpiryOCR:
    """
    Handles image preprocessing (deskewing) and Optical Character Recognition (OCR) 
    for cropped expiry date bounding boxes.
    """
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
        ocr_cfg = self.config.get("ocr", {})
        logger.info(f"Initializing EasyOCR with languages: {ocr_cfg.get('languages')}")
        self.reader = easyocr.Reader(
            ocr_cfg.get("languages", ["en"]), 
            gpu=ocr_cfg.get("use_gpu", True)
        )

    def _load_config(self, config_path: str) -> dict:
        if not os.path.exists(config_path):
            logger.warning(f"OCR config not found at {config_path}. Using defaults.")
            return {}
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Detects the text angle and rotates the image to make the text horizontal.
        Uses binary thresholding and minAreaRect for angle calculation.
        """
        if not self.config.get("preprocessing", {}).get("deskew", True):
            return image

        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding to invert the text (white text, black background)
        # assuming dark text on light background normally.
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find all non-zero points (text pixels)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return image # No text detected, return original
            
        # Get the minimum bounding rectangle enclosing all text pixels
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # OpenCV minAreaRect angle rules (cv2 >= 4.5)
        # The angle is returned in the range [0, 90)
        if angle > 45:
            angle = angle - 90

        limit = self.config.get("preprocessing", {}).get("deskew_angle_limit", 45)
        if abs(angle) > limit:
            logger.debug(f"Detected angle {angle:.2f} exceeds limit. Skipping rotation.")
            return image

        logger.debug(f"Deskewing image by {angle:.2f} degrees.")
        
        # Rotate the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Use BORDER_REPLICATE to avoid black borders
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def extract_text(self, image: np.ndarray) -> str:
        """Runs the OCR engine on the processed image."""
        # detail=0 returns only the text strings, not bounding boxes/confidences
        results = self.reader.readtext(image, detail=0, paragraph=False)
        
        # Join multiple detected text blocks into a single string
        final_text = " ".join(results).strip()
        return final_text

    def process(self, crop: np.ndarray) -> str:
        """Executes the full preprocessing and OCR pipeline on a cropped image."""
        straightened_crop = self.deskew(crop)
        text = self.extract_text(straightened_crop)
        return text