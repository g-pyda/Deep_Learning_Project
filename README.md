# DL-based framework for expiration date recognition
## Deep Learning project

### Used datasets:
- https://www.kaggle.com/datasets/kimhyeongminkhu/korean-expiry-date-ocr
- https://huggingface.co/datasets/dimun/ExpirationDate/tree/main

### Data preparation
Chosen datasets are merged into one YOLO adjusted dataset and additionally divided into test, train and validation sets. The config is constructed as follows:
```yaml
seed: 42                                                # seed used for random division
logging_path: "logs/dataset_preprocessor.log"           # path for logs saving
split_ratio:                                            # dataset division ratios
  train: 0.8
  val: 0.1
  test: 0.1

directories:                                            # target directories
  raw_dir: "data/raw"                                   # unprocessed datasets are saved here
  output_dir: "data/processed"                          # the ready processed dataset will be assembled here

datasets:                                               # list of merged datasets
  kaggle_korean:
    source_id: "kimhyeongminkhu/korean-expiry-date-ocr" # dataset name at kaggle / hface
    raw_path: "data/raw/kaggle_korean"                  # path to unprocessed dataset to download
    data_type: "kaggle_yolo_nested"                     # specification for different data loading procedures    
                                                        # possible values: "kaggle_yolo_nested", "hf_zip"
```

The data preprocessing is performed with the command:
```bash
python src/dataset_preprocessor.py --mode {all/preprocess/download} --config {path to config file}
```

### YOLO training
The datasets can be then used for training of the YOLO model. The configuration of the training process looks as follows:
```yaml
#YOLOv8 Architecture Configuration
logging_path: "logs/detector/train_detector_all.log"  # path for logs saving
model:
  weights: 'yolov8m.pt'                               # YOLO model to be trained

# Dataset Configuration
data:
  dataset_yaml: "data/processed/dataset.yaml"         # path to the processed dataset

# Training Hyperparameters
training:
  epochs: 200                                         # number of epochs
  batch_size: 32                                      # size of batch
  img_size: 640                                       # image scale
  optimizer: "Adam"                                   # optimizer (YOLO suppoorts SGD, Adam, AdamW, NAdam, RAdam, RMSProp)
  learning_rate: 0.001                                # learning rate
  patience: 10                                        # early stopping criterion
  device: "0"                                         # device specification (auto-detection if left empty)
  workers: 4                                          # number of dataloader workers
```


The training can be perfgormed with the specific command:
```bash
python src/train_detector.py --config config/detector_config.yaml
```

### OCR text detection
The last step is the use of the picture part detected by the YOLO for OCR text detection. It is configured with two config files:
```yaml
# Trained YOLOv8 configuration
model:
  weights: '/bakha/vhome/gpyda/scratch/Deep_Learning_Project/runs/detect/outputs/detector/run_20260620_170339/weights/best.pt'
```
```yaml
# OCR engine configuration
logging_path: "logs/ocr_pipe/pipeline_manager_all.log"  # path for log file
ocr:
  engine: "easyocr"                                     # OCR engine
  languages: ["en"]                                     # language specification
  use_gpu: true
  
# Image Processing Parameters
preprocessing:
  deskew: true
  deskew_angle_limit: 45                                # ignore angles larger than this (assume false positive rotation)
  padding: 5                                            # pixels to pad after deskewing
```

The date retrieval from the image is performed with the command:
```bash
python src/pipeline_manager.py 
    --det_config {path to YOLO detector configuration}
    --ocr_config {path to OCR configuration}
    --log_path {path to the logging file}
    --img {path to the tested picture}
```
and returns the read date (in string form) and the algorithm confidency.



