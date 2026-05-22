# DL-based framework for expiration date recognition
## Deep Learning project

### Used datasets:
- https://www.kaggle.com/datasets/ziya07/retail-food-package-expiry-date-dataset
- https://felizang.github.io/expdate/
- https://www.kaggle.com/datasets/kimhyeongminkhu/korean-expiry-date-ocr
- https://huggingface.co/datasets/dimun/ExpirationDate/tree/main

### Plan of work
#### 1. Construction of dataset pipeline

Proposed pipeline:
1. expiry date region detection with YOLO (only to separate the date from the rest of the picture)
2. data preparation: optional rotation, color improvement for better CNN work
3. date recognition with CNN or other model (maybe CRNN?)

Proposed directions of research:
- model training with/without data preprocessing
- search for optimal hyperparametrization, model architecture
- different date components recovery pipelines (whole date retrieval vs focus on DD/MM/YYYY components separately)
