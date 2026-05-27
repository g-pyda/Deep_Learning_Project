# DL-based framework for expiration date recognition
## Deep Learning project

### Used datasets:
<!-- - https://www.kaggle.com/datasets/ziya07/retail-food-package-expiry-date-dataset -->
<!-- - https://felizang.github.io/expdate/ -->
- https://www.kaggle.com/datasets/kimhyeongminkhu/korean-expiry-date-ocr
- https://huggingface.co/datasets/dimun/ExpirationDate/tree/main

### Data preparation
Chosen datasets are merged into one YOLO adjusted dataset and additionally divided into test, train and validation sets. The config is constructed as follows:
```yaml
seed: 42                                        # seed used for random division
logging_path: "logs/dataset_preprocessor.log"   # path for logs saving
split_ratio:                                    # dataset division ratios
  train: 0.8
  val: 0.1
  test: 0.1

directories:                    # target directories
  raw_dir: "data/raw"           # unprocessed datasets are saved here
  output_dir: "data/processed"  # the ready processed dataset will be assembled here

datasets:                       # list of merged datasets
  kaggle_korean:
    source_id: "kimhyeongminkhu/korean-expiry-date-ocr"                                     # dataset name at kaggle / hface
    raw_path: "data/raw/kaggle_korean"                                                      # path to unprocessed dataset to download
    data_type: "kaggle_yolo_nested"                                                         # specification for different data loading procedures    
                                                                                            # possible values: "kaggle_yolo_nested", "hf_zip"
```

The data preprocessing is performed with the command:
```bash
python src/dataset_preprocessor.py --mode all
python src/train_detector.py --config config/detector_config.yaml
python src/pipeline_manager.py --img data/processed/images/val/PICTURE_NAME.jpg
```

Proposed pipeline:
1. expiry date region detection with YOLO (only to separate the date from the rest of the picture)
2. data preparation: optional rotation, color improvement for better CNN work
3. date recognition with CNN or other model (maybe CRNN?)

Proposed directions of research:
- search for optimal hyperparametrization, model architecture
- different date components recovery pipelines (whole date retrieval vs focus on DD/MM/YYYY components separately)


