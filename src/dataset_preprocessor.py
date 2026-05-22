import os
import json
import yaml
import logging
import random
import argparse
import shutil
import zipfile
import csv
from pathlib import Path
from PIL import Image

# Importers for automatic dataset downloading
import kagglehub
from huggingface_hub import hf_hub_download

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("dataset_pipeline.log"),
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

# ==============================================================================
# DOWNLOAD MANAGER
# ==============================================================================

def download_datasets(config: dict):
    """Downloads and extracts datasets based on their configuration type."""
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
            if data_type in ['kaggle_csv', 'kaggle_yolo_nested']:
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

            elif data_type == 'github_csv':
                logger.info(f"[{name}] GitHub downloads disabled by user. Please ensure data is manually placed in {raw_path}.")
            
            else:
                logger.warning(f"[{name}] Unknown data_type: {data_type}")
                
        except Exception as e:
            logger.error(f"[{name}] Download failed: {e}")

# ==============================================================================
# PARSING LOGIC (TO UNIFIED FORMAT)
# ==============================================================================

def parse_kaggle_csv(ds_cfg: dict) -> list:
    dataset = []
    base_dir = Path(ds_cfg['raw_path'])
    csv_path = base_dir / ds_cfg['annotation_path']
    img_dir = base_dir / ds_cfg['images_dir']

    if not csv_path.exists():
        logger.warning(f"[kaggle_csv] Missing CSV file: {csv_path}")
        return dataset

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Adjust these column names based on the actual Kaggle CSV structure
                img_path = img_dir / row.get('filename', row.get('image', ''))
                if not img_path.exists():
                    continue
                
                xmin = float(row.get('xmin', 0))
                ymin = float(row.get('ymin', 0))
                xmax = float(row.get('xmax', 0))
                ymax = float(row.get('ymax', 0))
                text = row.get('text', '')

                dataset.append({
                    "original_path": str(img_path),
                    "bbox": [xmin, ymin, xmax, ymax],
                    "text": text,
                    "source": "kaggle_retail"
                })
    except Exception as e:
        logger.error(f"[kaggle_csv] Parsing error: {e}")

    return dataset

def parse_yolo_nested(ds_cfg: dict) -> list:
    dataset = []
    current_working_directory = os.getcwd()
    full_path = os.path.join(current_working_directory, ds_cfg['raw_path'])
    base_dir = Path(full_path)
    labels_dir = base_dir / "labels"
    
    print(f"Labels directory target: {labels_dir}")
    if labels_dir.exists():
        print(f"Contents of labels directory: {os.listdir(labels_dir)}")
    else:
        logger.error(f"Labels directory does not exist: {labels_dir}")
        return dataset
    
    # Recursively find all .txt files inside the labels folder structure
    for txt_path in labels_dir.rglob("*.txt"):
        # Safely convert path to string to use replace rules
        txt_path_str = str(txt_path)
        
        # Mirror the text path layout over to the images directory structure
        # Example conversion: .../labels/test/001.txt -> .../images/test/001.<ext>
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            possible_img_str = txt_path_str.replace(os.sep + 'labels' + os.sep, os.sep + 'images' + os.sep).replace('.txt', ext)
            possible_path = Path(possible_img_str)
            if possible_path.exists():
                img_path = possible_path
                break
                
        if not img_path:
            logger.warning(f"[yolo_nested] Image counterpart missing for label: {txt_path.name}")
            continue
            
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x_center, y_center, w, h = map(float, parts[1:5])
                        
                        # Convert normalized YOLO format back to absolute pixel coordinates
                        xmin = (x_center - w / 2.0) * img_w
                        ymin = (y_center - h / 2.0) * img_h
                        xmax = (x_center + w / 2.0) * img_w
                        ymax = (y_center + h / 2.0) * img_h
                        
                        dataset.append({
                            "original_path": str(img_path),
                            "bbox": [xmin, ymin, xmax, ymax],
                            "text": "",  # YOLO formats lack textual transcription properties
                            "source": "kaggle_korean"
                        })
        except Exception as e:
            logger.error(f"[yolo_nested] Error processing {txt_path}: {e}")
            
    logger.info(f"[yolo_nested] Parsed {len(dataset)} records from YOLO nested structure.")
    return dataset

def parse_hf_zip(ds_cfg: dict) -> list:
    """Parses the HuggingFace dimun/ExpirationDate dataset using its actual JSON structure."""
    dataset = []
    base_dir = Path(ds_cfg['raw_path'])
    json_path = base_dir / ds_cfg['annotation_path']
    img_dir = base_dir / ds_cfg['images_dir']

    if not json_path.exists():
        logger.warning(f"[hf_zip] Missing JSON file: {json_path}")
        return dataset

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # The JSON root is a dictionary where keys are image file names
        for filename, img_meta in data.items():
            img_path = img_dir / filename
            if not img_path.exists():
                logger.error(f"[hf_zip] Image file missing: {img_path}")
                continue
                
            # Iterate through the main annotations list
            annotations = img_meta.get("ann", [])
            for ann in annotations:
                # We only want the full expiration date bounding box
                if ann.get("cls") == "exp":
                    bbox = ann.get("bbox")  # Format: [xmin, ymin, xmax, ymax]
                    text = ann.get("transcription", "")
                    
                    if bbox and len(bbox) == 4:
                        dataset.append({
                            "original_path": str(img_path),
                            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                            "text": str(text),
                            "source": "hf_dimun"
                        })
                        
    except Exception as e:
        logger.error(f"[hf_zip] JSON parsing error: {e}")
    return dataset

def parse_github_csv(ds_cfg: dict) -> list:
    dataset = []
    base_dir = Path(ds_cfg['raw_path'])
    csv_path = base_dir / ds_cfg['annotation_path']

    if not csv_path.exists():
        return dataset

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = base_dir / row['image_filename']
                if not img_path.exists():
                    continue
                
                dataset.append({
                    "original_path": str(img_path),
                    "bbox": [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])],
                    "text": row['text'],
                    "source": "github_expdate"
                })
    except Exception as e:
        logger.error(f"[github_csv] CSV parsing error: {e}")

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

    # Generate YOLO dataset.yaml
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

    # 1. Download Mode
    if mode in ['download', 'all']:
        download_datasets(config)

    # 2. Preprocess Mode
    if mode in ['preprocess', 'all']:
        logger.info("=== Starting Parsing Phase ===")
        all_data = []
        datasets_cfg = config.get('datasets', {})

        for name, ds_cfg in datasets_cfg.items():
            data_type = ds_cfg.get('data_type')
            logger.info(f"Parsing {name}...")

            if data_type == 'kaggle_csv':
                all_data.extend(parse_kaggle_csv(ds_cfg))
            elif data_type == 'kaggle_yolo_nested':
                all_data.extend(parse_yolo_nested(ds_cfg))
            elif data_type == 'hf_zip':
                all_data.extend(parse_hf_zip(ds_cfg))
            elif data_type == 'github_csv':
                all_data.extend(parse_github_csv(ds_cfg))
        
        logger.info(f"Total unified annotations collected: {len(all_data)}")
        
        # 3. Export Data
        save_as_yolo(all_data, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for Expiry Date Datasets.")
    parser.add_argument("--config", type=str, default="config/data_config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=['download', 'preprocess', 'all'], 
        default='all',
        help="Action to perform: 'download' (only fetch data), 'preprocess' (only parse existing data), or 'all'."
    )
    args = parser.parse_args()
    
    main(args.config, args.mode)