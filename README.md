# 🌍 Land Use and Land Cover Change Detection (U-Net + TensorFlow)

A **deep-learning semantic segmentation** system for Land Use/Land Cover (LULC) using satellite imagery.  
The core model is a **U-Net** implemented in **TensorFlow/Keras**. It produces pixel-wise masks for classes
such as vegetation, water, urban/built-up, barren, etc., and then performs **change analysis** and **trend
visualization** across years/regions.

---

## 🔍 Problem Statement

Climate change and rapid urbanization alter land use patterns. This project:

- Segments satellite images into LULC classes with **U-Net (TensorFlow)**  
- Compares masks across time to quantify **area changes**  
- Visualizes **class distribution** and **temporal trends** per region

---

## 👤 My Role

> Team project (4 members) completed in pre final year.

- **Puneeth Hegde** – Data preprocessing, training the **U-Net (TensorFlow)**, pipeline integration, visualizations  
- **Sarvan D Suvarna** – Dataset collection & augmentation  
- **Shamith Vakwadey** – Experiment design & hyperparameter tuning  
- **Abhishek M** – Results analysis & documentation

---

## 🗂️ Project Structure

```

land-use-land-cover/
├── scripts/
│   ├── preprocess.py        # Tiling/normalization/splits
│   ├── segment\_images.py    # Inference with trained U-Net
│   ├── generate\_masks.py    # Utilities for mask preparation (if needed)
│   ├── analyze.py           # Area stats & change detection across years
│   ├── rename\_images.py     # Helper for dataset cleanup
│   ├── train\_unet.py        # Builds & trains U-Net in TensorFlow/Keras
│   ├── dash1.py             # Interactive dashboard (Dash)
│   └── utils/               # Common helpers
├── main.py                  # End-to-end runner (defaults)
├── requirements.txt
├── LICENSE
└── README.md

````

---

## 🧠 Model: U-Net (TensorFlow/Keras)

- **Architecture:** Encoder–decoder with skip connections (U-Net)  
- **Input size:** configurable (e.g., 256×256)  
- **Loss/metrics:** Cross-Entropy with **Dice/IoU** metrics (configurable in `train_unet.py`)  
- **Augmentation:** random flips/rotations/crops (set inside the training script)  
- **Outputs:** multi-class mask per image (one channel per class)

> Set `NUM_CLASSES`, `IMG_SIZE`, and paths at the top of `train_unet.py` (or via CLI args if provided).

---

## 🚀 Getting Started

### 1) Clone & Install
```bash
git clone https://github.com/puneeth-hegde/land-use-land-cover
cd land-use-land-cover
pip install -r requirements.txt
````

### 2) Prepare Data

Organize imagery and masks (adjust names to your dataset):

```
data/
├── raw/                      # original rasters by region/year
├── images/                   # tiles for training/validation
├── masks/                    # corresponding label masks
└── predictions/              # output masks from inference
```

Use the helpers as needed:

```bash
python scripts/preprocess.py        # tile/normalize/split
python scripts/generate_masks.py    # if you need to build masks from labels
python scripts/rename_images.py     # optional cleanup
```

### 3) Train the U-Net

```bash
python scripts/train_unet.py
```

This saves the best model (e.g., `models/unet_best.h5`) and logs **IoU/Dice/accuracy**.

### 4) Run Inference (Segmentation)

```bash
python scripts/segment_images.py --model models/unet_best.h5 \
  --input data/images/test --out data/predictions
```

### 5) Change Analysis & Dashboard

```bash
python scripts/analyze.py  # computes class areas & deltas across years/regions
python scripts/dash1.py    # launches Dash dashboard (localhost)
```

> Or run everything with defaults:

```bash
python main.py
```

---

## 📊 Outputs

* Per-image **segmentation masks** (PNG/GeoTIFF depending on config)
* **Class-wise area** summaries per region/year
* **Change maps** and **trend charts** across time
* Interactive **Dash dashboard** for exploration

---

## 🧩 Requirements

Key packages (see `requirements.txt` for exact versions):

* **tensorflow** / **keras**
* numpy, pandas, scikit-image, opencv-python
* matplotlib, plotly, dash
* (optional) rasterio/geopandas for geospatial workflows

Install:

```bash
pip install -r requirements.txt
```

---

## 📄 License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

U-Net architecture by Ronneberger et al. (2015). Inspired by common remote-sensing LULC practices.
