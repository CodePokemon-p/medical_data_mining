# Breast Cancer Detection using Medical Image Data Mining

## Project Overview
A comprehensive data mining pipeline for automated breast cancer detection using mammography images. This project implements a complete workflow from data gathering to machine learning model deployment, focusing on clinical applicability and research reproducibility.

## Key Features
- ✅ Complete Data Mining Pipeline - From raw DICOM to insights  
- ✅ Multiple ML Algorithms - Classification, Clustering, Binary Detection  
- ✅ Medical Domain Expertise - BI-RADS, breast density, clinical annotations  
- ✅ Production-Ready Code - Modular, well-documented, reproducible  
- ✅ Interactive Visualizations - ROC curves, confusion matrices, feature importance  

## Dataset Information

| Metric        | Value                       | Description                               |
|---------------|----------------------------|-------------------------------------------|
| Dataset       | VinDr-Mammo                | Breast cancer screening dataset           |
| Size          | 3.34 GB                    | Compressed DICOM + PNG masks              |
| Images        | 1,488                      | Mammograms with expert annotations        |
| Format        | DICOM + CSV                | Medical imaging standard                  |
| Annotations   | Bounding boxes, BI-RADS, finding categories | Expert radiologist labels |
| Source        | Kaggle                     | Publicly available                        |

## Project Structure
breast-cancer-detection/
├── data/
│   ├── archive1/
│   │   ├── manifest.xlsx
│   │   ├── finding_annotations.xlsx
│   │   ├── images/
│   │   └── masks/
│   ├── processed/
│   │   ├── merged_dataset.csv
│   │   ├── train_set.csv
│   │   ├── validation_set.csv
│   │   └── test_set.csv
│   └── splits/
├── notebooks/
│   ├── 01_data_gathering.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_data_splitting.ipynb
│   ├── 04_algorithm_application.ipynb
│   └── Medical_Image_Data_Mining_COMPLETE.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── models.py
│   ├── utils.py
│   └── config.py
├── results/
│   ├── figures/
│   ├── metrics/
│   └── reports/
├── docs/
│   ├── project_report.pdf
│   ├── viva_preparation.md
│   └── clinical_guidelines.md
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md

## Quick Start

### Option 1: Google Colab (Recommended)
Click the badge to open Colab:  
[Open In Colab](https://colab.research.google.com/)

1. Mount your Google Drive  
2. Upload dataset to `/content/drive/MyDrive/data/archive1/`  
3. Run all notebook cells  

### Option 2: Local Installation
# 1. Clone repository
git clone https://github.com/your-username/breast-cancer-detection.git
cd breast-cancer-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# From Kaggle: https://www.kaggle.com/datasets/...
# Extract to data/archive1/

# 5. Run complete notebook
jupyter notebook notebooks/Medical_Image_Data_Mining_COMPLETE.ipynb

## Requirements
python==3.8+
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
pydicom==2.3.1
matplotlib==3.7.1
seaborn==0.12.2
opencv-python==4.8.1
jupyter==1.0.0
notebook==6.5.4
pillow==10.0.0
scikit-image==0.21.0
tqdm==4.65.0
prettytable==3.8.0

## Data Mining Pipeline

Phase 1: Data Gathering
manifest = pd.read_excel("data/archive1/manifest.xlsx")
findings = pd.read_excel("data/archive1/finding_annotations.xlsx")
print(f"Dataset loaded: {len(manifest)} images, {len(findings)} annotations")

Phase 2: Data Preparation
- ✅ Extract image IDs from DICOM paths  
- ✅ Handle 18,357+ null values  
- ✅ Data type conversion & normalization  
- ✅ Feature engineering (breast density encoding)  

Phase 3: Data Transformation
- ✅ Train/Validation/Test split (70/15/15)  
- ✅ Stratified sampling by finding categories  
- ✅ Patient-level splitting (prevents data leakage)  

Phase 4: Algorithm Application
Algorithm       | Type        | Accuracy | AUC   | Use Case
----------------|------------|---------|-------|---------------------------
Random Forest   | Multi-class | 78%     | -     | Finding classification
K-Means         | Clustering  | -       | -     | Pattern discovery
Random Forest   | Binary      | 69.1%   | 0.694 | Mass detection
Logistic Reg.   | Binary      | 62.8%   | 0.594 | Baseline comparison
SVM             | Binary      | 66.7%   | 0.404 | High-dimensional data

## Results & Performance
- Multi-Class Classification: Best Model: Random Forest, Accuracy: 78%, Key Feature: Breast density  
- Mass Detection (Binary): Best Model: Random Forest, Accuracy: 69.1%, AUC: 0.694  
- Clustering Analysis: Algorithm: K-Means, Optimal Clusters: 4, Interpretation: Breast density patterns  

## Clinical Relevance
- Automated Triage System  
- Second Opinion Tool  
- Quality Assurance  
- Education for radiologists  

## Technical Implementation
- Modular Python scripts in src/  
- Preprocessing with pydicom  
- ML models with scikit-learn  
- Visualizations: confusion matrix, ROC, feature importance  

## Contributing
- Fork the repository  
- Create a branch (`git checkout -b feature/YourFeature`)  
- Commit changes (`git commit -m "Add feature"`)  
- Push branch (`git push origin feature/YourFeature`)  
- Open Pull Request  

## License
MIT License – see LICENSE file for details  

## Acknowledgments
- VinDr-Mammo dataset providers  
- Radiologists for annotations  
- Open source community: Pandas, Scikit-learn, Pydicom  

## Contact
- LinkedIn: https://www.linkedin.com/in/sabasaleempk
- Email: s22bseen1m01042@iub.edu.pk
