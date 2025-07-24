# 🌍 Land Use and Land Cover Change Detection

A Deep Learning–based system to analyze and predict Land Use and Land Cover (LULC) changes over time using satellite images. This project helps in understanding environmental trends like deforestation, urbanization, and water body shrinkage by leveraging segmentation, analysis, and forecasting models.

---

## 🔍 Problem Statement

Climate change, urban expansion, and human activities drastically alter land use patterns. This project aims to:

- Detect various geographical aspects (e.g., forests, water bodies, barren land)
- Track changes across multiple years
- Predict future LULC distribution using machine learning models

---

## 👤 My Role

> This project was completed as a team of 4 during our final year engineering project.

- **Puneeth Hegde** – Data preprocessing, pipeline building, visualization, final integration  
- **Sarvan D Suvarna** – Dataset collection and augmentation  
- **Shamith Vakwadey** – Model training and tuning  
- **Abhishek M** – Results analysis and documentation

---

## 🗂️ Project Structure

```
land_use_project/
├── data/
│   ├── raw/                 # Raw NLCD images in TIFF format
│   ├── processed/           # Preprocessed images (resized, sorted)
│   └── analysis_results/    # Calculated area statistics per region
│
├── models/
│   └── trend_predictor.pkl  # Trained model for forecasting changes
│
├── scripts/
│   ├── preprocess.py        # Cleans & prepares data
│   ├── analyze.py           # Segments land cover types & calculates area
│   ├── train_predictor.py   # Trains forecasting model
│   ├── dashboard.py         # Builds interactive dashboard (Dash)
│   └── utils/
│       ├── metadata_utils.py
│       └── image_utils.py
│
├── main.py                  # One-click full pipeline execution
└── requirements.txt         # All required packages
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/puneeth-hegde/land-use-land-cover
cd land-use-land-cover
```

### 2. Install Dependencies

Make sure you have **Python 3.x** installed.

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset

Organize your NLCD TIFF images by region and year:

```
data/
└── raw/
    ├── region_1/
    ├── region_2/
    └── region_3/
```

Each region folder should contain TIFF images sorted by year.

---

### 4. Run the Full Pipeline

```bash
python main.py
```

Or run steps individually:

```bash
python scripts/preprocess.py        # Prepare & resize images
python scripts/analyze.py           # Detect & quantify features
python scripts/train_predictor.py   # Train ML model
python scripts/dashboard.py         # Launch dashboard (localhost)
```

---

## 📊 Output

- 📌 Pie charts of land types per region/year  
- 📈 Time-series graphs of land cover trends  
- 📍 Forecasted LULC map for the next 10 years  
- 🖥️ Interactive real-time dashboard built using Dash

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`, including:

- numpy, pandas, matplotlib  
- opencv-python, geopandas  
- scikit-learn  
- dash, plotly  

To install them:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
