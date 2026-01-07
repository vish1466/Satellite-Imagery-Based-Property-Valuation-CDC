# Satellite-Imagery-Based-Property-Valuation-CDC

### Multimodal House Price Prediction using Satellite Imagery & Tabular Data

Overview : 
This project predicts house prices by combining structured tabular features with satellite imagery using a multimodal deep learning approach. In addition to prediction, the model is made interpretable using Grad-CAM, which highlights regions of satellite images that influence pricing decisions.

The project explores:
- Traditional tabular-only modeling (XGBoost)
- A CNN + MLP multimodal architecture
- Visual explainability via Grad-CAM

## Setup & Installation

### Clone the Repository
```bash
git clone https://github.com/vish1466/Satellite-Imagery-Based-Property-Valuation-CDC.git
cd Satellite-Imagery-Based-Property-Valuation-CDC
```

### Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies 
```bash
pip install -r requirements.txt
```
Key libraries used:
-PyTorch & Torchvision
-XGBoost
-Scikit-learn
-OpenCV, cv2
-Pandas, NumPy
-Matplotlib, Seaborn
- mercantile, requests

## How to Run the Project
1. Exploratory Data Analysis
```bash
eda/eda.ipynb
```
2️. Feature Engineering & Preprocessing
```bash
preprocessing/feature_engineering.ipynb
```
3. Train Tabular Baseline (XGBoost)
```bash
models/xgb_tabular.ipynb
```
4. Train Multimodal Model (CNN + MLP)
```bash
models/multimodal_training.ipynb
```
- CNN: Pretrained ResNet18
- MLP: Tabular feature encoder
- Fusion: Concatenation + regression head

5. Grad-CAM Explainability
```bash
models/grad_cam.ipynb
```
To generate heatmaps, which highlights the regions that influenced the price

## Hardware Requirements
- CPU (supported)
- GPU (Google Colab Recommended)

### Test Set Predictions
Test set predictions are stored in the below file. These are the values that are obtained by providing the test dataset as input for the trained Multimodal.
```bash
22118037_final.csv
```

