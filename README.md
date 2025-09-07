# 🦴 Bone Fracture Detection using YOLOv11m

This repository contains the implementation of a **Bone Fracture Detection** model trained on the [[Human Bone Fractures Image Dataset](https://www.kaggle.com/datasets/jockeroika/human-bone-fractures-image-dataset/data)].  
The project leverages the **YOLOv11m** architecture from [[Ultralytics](https://www.ultralytics.com/)] for object detection and localization of fractures in medical images.  

---

## 📌 Dataset
- **Source**: [[Human Bone Fractures Image Dataset](https://www.kaggle.com/datasets/jockeroika/human-bone-fractures-image-dataset/data)]  
- The dataset consists of multi-modal X-ray and CT images annotated with bounding boxes around fracture regions.  
- Categories represent different types of bone fractures.  

[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/jockeroika/human-bone-fractures-image-dataset/data)

---

## ⚙️ Model Training
- **Base Model**: `yolo11m.pt` (pretrained on COCO)  
- **Training Parameters**:  
  - `epochs = 30`  
  - `imgsz = 640`  
  - `device = 0` (GPU)  
- Config file used: `data.yaml` from the dataset.  

Training was conducted in Google Colab with T4 GPU acceleration.

---

## 📊 Results
During training, YOLOv11m optimized the following loss components:  
- **box_loss** → Bounding box regression  
- **cls_loss** → Classification loss  
- **dfl_loss** → Distribution Focal Loss  

Evaluation Metrics:  
- **mAP50-95: 0.4459**
- **mAP50: 0.8690**
- **Precision: 0.8560**
- **Recall: 0.8282**
- **F1-score:0.8204**
---

## 🚀 Inference
To run inference with the trained model:  

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("best.pt")

# Run inference on an image
results = model.predict("sample_image.jpg", save=True, imgsz=640, conf=0.25)
```

The predicted image with bounding boxes will be saved automatically.

🔬 Research Contribution

This work demonstrates the feasibility of using YOLOv11m for medical image analysis, specifically for bone fracture detection.
Future directions may include:
- Fine-tuning with larger medical datasets
- Incorporating explainability (Grad-CAM, attention maps)
- Comparing YOLOv11m with transformer-based models
