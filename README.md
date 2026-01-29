Below is a ready-to-paste `README.md` that matches what your notebooks + dissertation describe (Mask R-CNN via Detectron2, COCO-style annotations from masks, optional Google Earth Engine export + time-series area tracking). It **does not** mention an MSc or university context.  ([GitHub][1])

````md
# Water Body Segmentation & Change Tracking (Mask R-CNN / Detectron2)

Detect and segment surface water in satellite imagery using **Mask R-CNN (Detectron2)**, then **track water extent over time** by running inference on time-lapse imagery/videos (e.g., Landsat/Sentinel exports from Google Earth Engine).

This repo includes:
- A training pipeline that converts binary water masks into **COCO-style instance annotations**, trains a **Mask R-CNN R50-FPN** model, and evaluates it.
- An application pipeline that exports imagery time series (optional), runs **frame-by-frame inference**, and outputs **segmented video + per-frame water-pixel area + yearly trend plots**.

---

## Contents
- [Highlights](#highlights)
- [Repository Layout](#repository-layout)
- [Setup](#setup)
- [Data](#data)
- [Training & Evaluation](#training--evaluation)
- [Time-Series Application (Video / GEE)](#time-series-application-video--gee)
- [Outputs](#outputs)
- [Notes & Limitations](#notes--limitations)
- [References](#references)

---

## Highlights
- **Model:** Mask R-CNN (Detectron2) configured for **single-class** segmentation (water).
- **Annotation:** Convert paired `(image, mask)` datasets into **COCO JSON** (bbox + polygon segmentation derived from masks).
- **Time series:** Export imagery sequence (optional) → run inference per frame → compute **predicted water pixels** as a proxy for surface area → aggregate trends by year.

---

## Repository Layout
> Adjust to match your actual repo. This structure is recommended so paths don’t stay “Colab-only”.

```text
.
├─ notebooks/
│  ├─ Create_annotations_and_train_model_Final.ipynb
│  └─ Application.ipynb
├─ data/
│  ├─ raw/
│  │  ├─ images/
│  │  └─ masks/
│  └─ processed/
│     ├─ images_512/
│     ├─ masks_512/
│     ├─ coco_dataset.json
│     ├─ annotations_train.json
│     └─ annotations_test.json
├─ weights/
│  └─ model_final.pth
├─ outputs/
│  ├─ predictions/
│  ├─ videos/
│  ├─ csv/
│  └─ plots/
└─ README.md
````

---

## Setup

### Option A: Google Colab (recommended)

Both notebooks were written with Colab + Google Drive in mind. Open the notebooks in Colab and follow the cells in order.

### Option B: Local (Linux recommended)

**Detectron2** installation can vary by CUDA/PyTorch version. Follow Detectron2’s official install guidance first.

Basic outline:

1. Create an environment with PyTorch (matching your CUDA driver).
2. Install Detectron2 (from source is common):

   ```bash
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```
3. Install common deps:

   ```bash
   pip install opencv-python pillow numpy pandas matplotlib tqdm pycocotools scikit-image
   ```

---

## Data

### Training dataset (example)

The training notebook assumes a paired dataset of satellite images + corresponding binary masks (water=white, background=black). One commonly used public dataset is:

* Kaggle: **“Satellite Images of Water Bodies”** (Images + Masks)

Download it, then place:

* images → `data/raw/images/`
* masks  → `data/raw/masks/`

> The pipeline resizes imagery to **512×512** before generating annotations.

---

## Training & Evaluation

Open: `notebooks/Create_annotations_and_train_model_Final.ipynb`

What it does (high level):

1. **Resize** images & masks to a fixed resolution (e.g., `512×512`).
2. **Create COCO annotations** from masks:

   * bounding boxes from mask extents
   * segmentation polygons from mask contours
3. **Split** into train/test COCO JSON.
4. **Register datasets** in Detectron2.
5. **Train** Mask R-CNN using:

   * `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`
   * single class: `water`
6. **Evaluate** using COCO metrics (AP / IoU, etc.)
7. (Optional) Save example **predicted masks**, compute basic per-image metrics, export CSV summaries.

### Expected edit points

* Input paths (raw/processed)
* Output directory: `cfg.OUTPUT_DIR`
* Number of classes: `cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1`
* Score threshold: `cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5`

---

## Time-Series Application (Video / GEE)

Open: `notebooks/Application.ipynb`

This notebook supports an end-to-end flow:

### 1) (Optional) Export a time series video from Google Earth Engine

* Authenticate + initialize Earth Engine
* Pick:

  * `longitude`, `latitude`
  * `bufferzone` (meters)
  * satellite source (e.g., Landsat collection(s) for long history, Sentinel-2 for higher resolution)
  * date range
* Export to Google Drive as MP4 (RGB image collection required)

> You must replace any hard-coded project IDs and ensure your Earth Engine + Drive permissions are set up.

### 2) Run inference on the exported video

* Load `weights/model_final.pth` (or your chosen path)
* Run Detectron2 predictor per frame
* Draw masks/overlays and write:

  * **segmented output video**
  * **CSV** with per-frame `predicted_water_pixels`
* Aggregate by year and save **trend plots**

---

## Outputs

Depending on what you run, you’ll produce:

* `outputs/videos/`

  * segmented video overlays (mask drawn per frame)
* `outputs/csv/`

  * per-frame predicted water pixels
  * optional per-image evaluation summaries
* `outputs/plots/`

  * predicted water pixels aggregated by year (trend line)
* `outputs/predictions/`

  * predicted mask images / visualizations

---

## Notes & Limitations

* This is **single-class** segmentation: it’s trained to find *water vs. not-water*.
* Performance depends heavily on **image resolution** and scene characteristics:

  * large/open water works better than tiny/fragmented water features
  * clouds/shadows/turbidity can confuse segmentation
* Training is GPU-heavy; inference can run on CPU but will be slow for long videos.
* If you change image resolution, regenerate annotations (COCO polygons/bboxes depend on size).

---

## References

* He et al., *Mask R-CNN* (2017/2018)
* Detectron2 (Facebook AI Research / Meta)
* Kaggle: “Satellite Images of Water Bodies” dataset

---


