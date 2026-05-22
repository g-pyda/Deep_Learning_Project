import os
import yaml
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path

# Import our custom wrapper
from models.date_detector import DateDetector

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file does not exist: {config_path}")
        raise FileNotFoundError(f"Missing file: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main(config_path: str):
    logger.info("=== Starting Detector Pipeline ===")
    
    # 1. Load Configuration
    config = load_config(config_path)
    
    # 2. Setup Output Directories (run_[TIMESTAMP])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = Path("outputs/detector")
    run_name = f"run_{timestamp}"
    run_dir = project_dir / run_name
    
    # Create the base project directory if it doesn't exist
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Save a copy of the configuration to the run folder for reproducibility
    # Wait until Ultralytics creates the run folder, or create it ourselves beforehand
    run_dir.mkdir(parents=True, exist_ok=True)
    config_backup_path = run_dir / "detector_config.yaml"
    with open(config_backup_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False)
    logger.info(f"Configuration saved to {config_backup_path}")

    # 4. Initialize Model
    weights = config.get("model", {}).get("weights", "yolov8n.pt")
    detector = DateDetector(weights_path=weights)

    # 5. Extract Training Parameters
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    
    dataset_yaml = data_cfg.get("dataset_yaml", "data/processed/dataset.yaml")
    
    # Ensure absolute path for dataset_yaml to avoid YOLO execution path issues
    dataset_yaml_abs = os.path.abspath(dataset_yaml)
    if not os.path.exists(dataset_yaml_abs):
        logger.error(f"Dataset YAML not found at: {dataset_yaml_abs}")
        raise FileNotFoundError(f"Dataset YAML missing. Run preprocessor first.")

    # 6. Start Training
    # We pass 'project' and 'name' so ultralytics natively saves inside outputs/detector/run_[TIMESTAMP]
    # 'best.pt' and 'last.pt' will automatically be stored in run_dir/weights/
    detector.train(
        data=dataset_yaml_abs,
        epochs=train_cfg.get("epochs", 50),
        batch=train_cfg.get("batch_size", 16),
        imgsz=train_cfg.get("img_size", 640),
        optimizer=train_cfg.get("optimizer", "auto"),
        lr0=train_cfg.get("learning_rate", 0.001),
        patience=train_cfg.get("patience", 10),
        device=train_cfg.get("device", ""),
        workers=train_cfg.get("workers", 8),
        project=str(project_dir),
        name=run_name,
        exist_ok=True # Prevents YOLO from appending numbers (run_TIMESTAMP2) if it already exists
    )

    logger.info(f"=== Training Completed ===")
    logger.info(f"Best weights and metrics are saved in: {run_dir}/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Expiry Date Bounding Box Detector.")
    parser.add_argument("--config", type=str, default="config/detector_config.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    
    main(args.config)