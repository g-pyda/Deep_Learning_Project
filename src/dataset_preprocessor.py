import os
import json
import yaml
import logging
import random
import argparse
import shutil
import zipfile
from pathlib import Path
from PIL import Image

import kagglehub
from huggingface_hub import hf_hub_download

# 1. Default fallback logger (before config is loaded)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def setup_logging(log_path: str):
    """Configures logging to save to the specified file and output to console."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
    # force=True overwrites the default logger we set at the top of the file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        force=True 
    )
    logger.info(f"Logging successfully configured. Saving logs to: {log_path}")

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.error(f"Configuration file does not exist: {config_path}")
        raise FileNotFoundError(f"Missing file: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ==============================================================================
# DOWNLOAD MANAGER
# ==============================================================================

def download_datasets(config: dict):
    logger.info("=== Starting Download Phase ===")
    datasets_cfg = config.get('datasets', {})
    
    for name, ds_cfg in datasets_cfg.items():
        raw_path = Path(ds_cfg['raw_path'])
        data_type = ds_cfg.get('data_type')
        source_id = ds_cfg.get('source_id')
        
        if raw_path.exists() and any(raw_path.iterdir()):
            logger.info(f"[{name}] Data already exists at {raw_path}. Skipping download.")
            continue
            
        raw_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{name}] Downloading dataset ({data_type})...")

        try:
            if data_type == 'kaggle_yolo_nested':
                cache_path = kagglehub.dataset_download(source_id)
                shutil.copytree(cache_path, raw_path, dirs_exist_ok=True)
                logger.info(f"[{name}] Successfully copied from Kaggle cache to {raw_path}")

            elif data_type == 'hf_zip':
                file_name = ds_cfg.get('file_name')
                zip_path = hf_hub_download(repo_id=source_id, filename=file_name, repo_type="dataset")
                logger.info(f"[{name}] Extracting ZIP file...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(raw_path)
                logger.info(f"[{name}] Successfully extracted to {raw_path}")
                
            else:
                logger.warning(f"[{name}] Unknown data_type: {data_type}")
                
        except Exception as e:
            logger.error(f"[{name}] Download failed: {e}")

# ==============================================================================
# PARSING LOGIC (TO UNIFIED FORMAT)
# ==============================================================================

def parse_yolo_nested(ds_cfg: dict) -> list:
    dataset = []
    current_working_directory = os.getcwd()
    full_path = os.path.join(current_working_directory, ds_cfg['raw_path'])
    base_dir = Path(full_path)
    
    if not base_dir.exists():
        logger.error(f"[yolo_nested] Base directory does not exist: {base_dir}")
        return dataset
    
    # Recursively find all .txt files inside any 'labels' folder across all sub-datasets
    for txt_path in base_dir.rglob("*.txt"):
        txt_path_str = str(txt_path)
        
        # Skip files that are not within a 'labels' directory
        if f"{os.sep}labels{os.sep}" not in txt_path_str:
            continue
            
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            possible_img_str = txt_path_str.replace(os.sep + 'labels' + os.sep, os.sep + 'images' + os.sep).replace('.txt', ext)
            possible_path = Path(possible_img_str)
            if possible_path.exists():
                img_path = possible_path
                break
                
        if not img_path:
            continue
            
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x_center, y_center, w, h = map(float, parts[1:5])
                        
                        xmin = (x_center - w / 2.0) * img_w
                        ymin = (y_center - h / 2.0) * img_h
                        xmax = (x_center + w / 2.0) * img_w
                        ymax = (y_center + h / 2.0) * img_h
                        
                        dataset.append({
                            "original_path": str(img_path),
                            "bbox": [xmin, ymin, xmax, ymax],
                            "text": "", 
                            "source": "kaggle_korean"
                        })
        except Exception as e:
            logger.error(f"[yolo_nested] Error processing {txt_path}: {e}")
            
    logger.info(f"[yolo_nested] Parsed {len(dataset)} records from YOLO nested structures.")
    return dataset

def parse_hf_zip(ds_cfg: dict) -> list:
    dataset = []
    base_dir = Path(ds_cfg['raw_path'])
    
    if not base_dir.exists():
        logger.warning(f"[hf_zip] Base directory does not exist: {base_dir}")
        return dataset

    # Recursively find all annotations.json files (handles train/ and evaluation/ automatically)
    for json_path in base_dir.rglob("annotations.json"):
        img_dir = json_path.parent / "images"
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for filename, img_meta in data.items():
                img_path = img_dir / filename
                if not img_path.exists():
                    continue
                    
                annotations = img_meta.get("ann", [])
                for ann in annotations:
                    if ann.get("cls") == "exp":
                        bbox = ann.get("bbox")
                        text = ann.get("transcription", "")
                        
                        if bbox and len(bbox) == 4:
                            dataset.append({
                                "original_path": str(img_path),
                                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                                "text": str(text),
                                "source": "hf_dimun"
                            })
        except Exception as e:
            logger.error(f"[hf_zip] JSON parsing error in {json_path}: {e}")

    logger.info(f"[hf_zip] Parsed {len(dataset)} records from HF JSON structures.")
    return dataset

# ==============================================================================
# YOLO EXPORT LOGIC
# ==============================================================================

def convert_bbox_to_yolo(size: tuple, bbox: list) -> tuple:
    img_width, img_height = size
    xmin, ymin, xmax, ymax = bbox

    xmin, xmax = max(0, xmin), min(img_width, xmax)
    ymin, ymax = max(0, ymin), min(img_height, ymax)

    x_center = ((xmin + xmax) / 2.0) / img_width
    y_center = ((ymin + ymax) / 2.0) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    return x_center, y_center, width, height

def save_as_yolo(all_data: list, config: dict):
    logger.info("=== Starting Export Phase (YOLOv8 Format) ===")
    
    total_records = len(all_data)
    if total_records == 0:
        logger.error("No data collected to save.")
        return

    random.seed(config.get('seed', 42))
    random.shuffle(all_data)

    splits = config.get('split_ratio', {'train': 0.8, 'val': 0.1, 'test': 0.1})
    train_end = int(total_records * splits['train'])
    val_end = train_end + int(total_records * splits['val'])

    splits_data = {
        'train': all_data[:train_end],
        'val': all_data[train_end:val_end],
        'test': all_data[val_end:]
    }

    out_dir = Path(config['directories']['output_dir'])
    if out_dir.exists():
        logger.warning(f"Output directory {out_dir} already exists. Cleaning up...")
        shutil.rmtree(out_dir)

    for split_name, data_split in splits_data.items():
        img_split_dir = out_dir / 'images' / split_name
        lbl_split_dir = out_dir / 'labels' / split_name
        img_split_dir.mkdir(parents=True, exist_ok=True)
        lbl_split_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for i, item in enumerate(data_split):
            orig_path = Path(item["original_path"])
            base_filename = f"{item['source']}_{split_name}_{i:06d}"
            new_img_path = img_split_dir / f"{base_filename}{orig_path.suffix}"
            new_lbl_path = lbl_split_dir / f"{base_filename}.txt"
            
            try:
                with Image.open(orig_path) as img:
                    img_size = img.size
                
                x_center, y_center, width, height = convert_bbox_to_yolo(img_size, item["bbox"])
                
                with open(new_lbl_path, 'w', encoding='utf-8') as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                shutil.copy2(orig_path, new_img_path)
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving {orig_path}: {e}")

        logger.info(f"Saved {saved_count} records to split '{split_name}'.")

    yaml_content = {
        'path': str(out_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'expiry_date'}
    }
    with open(out_dir / 'dataset.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True)
    logger.info(f"YOLO configuration saved to {out_dir / 'dataset.yaml'}")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main(config_path: str, mode: str):
    config = load_config(config_path)

    # Setup logging based on config path
    log_path = config.get('logging_path', 'logs/default.log')
    setup_logging(log_path)

    if mode in ['download', 'all']:
        download_datasets(config)

    if mode in ['preprocess', 'all']:
        logger.info("=== Starting Parsing Phase ===")
        all_data = []
        datasets_cfg = config.get('datasets', {})

        for name, ds_cfg in datasets_cfg.items():
            data_type = ds_cfg.get('data_type')
            logger.info(f"Parsing {name}...")

            if data_type == 'kaggle_yolo_nested':
                all_data.extend(parse_yolo_nested(ds_cfg))
            elif data_type == 'hf_zip':
                all_data.extend(parse_hf_zip(ds_cfg))
        
        logger.info(f"Total unified annotations collected: {len(all_data)}")
        
        save_as_yolo(all_data, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for Expiry Date Datasets.")
    parser.add_argument("--config", type=str, default="config/data_config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=['download', 'preprocess', 'all'], 
        default='all',
        help="Action to perform: 'download', 'preprocess', or 'all'."
    )
    args = parser.parse_args()
    
    main(args.config, args.mode)