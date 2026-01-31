# fire-detection

Fire detection software for a Machine Learning class project. The repo contains **two pipelines**: a **CNN image-classification** pipeline (four classes: fire, smoke, nothing, fire_and_smoke) and an **FCOS object-detection** pipeline (fire and smoke bounding boxes). Both use the **D-Fire** dataset ([D-Fire: an image dataset for fire and smoke detection](https://github.com/gaia-solutions-on-demand/DFireDataset)) — 21,000+ images with YOLO-format annotations from [Gaia, solutions on demand (GAIA)](https://github.com/gaia-solutions-on-demand/DFireDataset).

---

## Repository structure

All code and outputs are organized by role. **Run commands from the repository root.**

```
fire-detection/
├── main.py                    # Entry point: CNN training & evaluation
├── requirements.txt           # Python dependencies
│
├── src/                       # CNN classification pipeline (TensorFlow/Keras)
│   ├── models/
│   │   └── cnn.py             # CNN model definition and configs
│   ├── data/
│   │   ├── load_data.py       # tf.data loaders from Phase 1 CSVs
│   │   ├── phase1_classification_preprocess.py   # D-Fire → classification dataset
│   │   └── phase2_detection_metadata.py          # D-Fire → detection JSON for FCOS
│   ├── evaluation/
│   │   └── evaluate.py        # Metrics, logging, training plots
│   └── experiments/
│       └── experiments.py     # Experiment configs (baseline, dropout, etc.)
│
├── notebooks/
│   └── sample_selection.ipynb # Exploratory / sample-selection notebook
│
├── results/                   # CNN experiment outputs
│   └── experiment_results.json
│
├── fcos/                      # FCOS detection pipeline (PyTorch)
│   ├── fcos.py                # FCOS model
│   ├── fcos_dataset.py        # Dataset from Phase 2 JSON
│   ├── fcos_training.ipynb    # Training notebook (local + Colab)
│   └── __init__.py
│
├── models/                    # Other baselines (e.g. decision tree, KNN, RF)
│   └── ...
│
├── processed_dfire/           # Created by Phase 1: resized images + CSVs
├── D-Fire/                    # Raw D-Fire dataset (YOLO layout)
└── plots/                     # Training/confusion plots (optional)
```

### What lives where

| Path | Purpose |
|------|--------|
| **`main.py`** | Run CNN experiments: loads data via `src.data.load_data`, builds models from `src.models.cnn`, trains and evaluates, logs to `results/`. |
| **`src/models/cnn.py`** | CNN architecture and default `MODEL_CONFIG` / `COMPILE_CONFIG` / `TRAIN_CONFIG`. |
| **`src/data/`** | **Phase 1** turns D-Fire into a classification dataset (square images + per-split CSVs). **Phase 2** turns D-Fire into one detection JSON. **`load_data.py`** reads the Phase 1 CSVs and builds `tf.data.Dataset`s for `main.py`. |
| **`src/evaluation/evaluate.py`** | Evaluation metrics, `log_results()` (writes `results/experiment_results.json`), and `plot_training_history()`. |
| **`src/experiments/experiments.py`** | Dict of named experiments (baseline, deeper_network, dropout variants, etc.) used by `main.py` and `evaluate.py`. |
| **`notebooks/`** | Jupyter notebooks; open from repo root so `src` is on the path. |
| **`results/`** | CNN run logs; default file is `results/experiment_results.json`. |
| **`fcos/`** | FCOS detector: model, dataset (from Phase 2 JSON), and training notebook. |

---

## Quick start

1. **Install dependencies** (from repo root):
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the D-Fire dataset** in YOLO layout under `D-Fire/` (e.g. `D-Fire/train/images`, `D-Fire/train/labels`, and optionally `test/`, `validation/`). Download images and labels (or the pre-split train/val/test sets) from the [D-Fire GitHub repo](https://github.com/gaia-solutions-on-demand/DFireDataset).

3. **CNN classification pipeline**
   - Run Phase 1 preprocessing (writes `processed_dfire/` and CSVs), then run training:
   ```bash
   python -m src.data.phase1_classification_preprocess --dataset-dir D-Fire --output-dir processed_dfire --splits train validation test
   python main.py
   ```
   - Use `python main.py <experiment_name>` to run a single experiment (e.g. `baseline`, `deeper_network`).

4. **FCOS detection pipeline**
   - Run Phase 2 to build the detection JSON, then train via the notebook:
   ```bash
   python -m src.data.phase2_detection_metadata --dataset-dir D-Fire --output dfire_detection_annotations.json --splits train test validation
   ```
   - Open `fcos/fcos_training.ipynb` and run all cells (works locally or on Colab; see below).

---

## Phase 1 — Classification dataset

**Goal:** Turn D-Fire (YOLO) into a four-class classification dataset: square RGB images plus per-split metadata CSVs. Outputs are used by `src/data/load_data.py` and `main.py`.

**Script:** `src/data/phase1_classification_preprocess.py` (run as module).

```bash
python -m src.data.phase1_classification_preprocess \
  --dataset-dir D-Fire \
  --output-dir processed_dfire \
  --splits train validation test \
  --val-percent 10 \
  --resample-percent 30 \
  --log-every 25
```

**Options:**

- `--dataset-dir`: D-Fire root (e.g. `train/images`, `train/labels`).
- `--output-dir`: Where to write processed PNGs and CSVs (e.g. `processed_dfire`).
- `--splits`: Splits to produce. If `validation` (or `val`) is missing, it is derived from train using `--val-percent`.
- `--val-percent`: Fraction of (resampled) train used for validation when deriving it.
- `--image-size` / `--crop-strategy`: Size and crop strategy for square images.
- `--positive-class`: YOLO class id for "fire" (default 1); others → smoke; no labels → nothing; fire+smoke → fire_and_smoke.
- `--resample-percent` / `--resample-seed`: Optional down-sampling while keeping class ratios.
- `--skip-existing`: Skip images already present in the output dir.

**Outputs:**

- `processed_dfire/<split>/{fire,smoke,nothing,fire_and_smoke}/*.png`
- `processed_dfire/<split>_metadata.csv` (columns: `image_id`, `split`, `original_path`, `processed_path`, `label_idx`, `label_name`, `num_boxes`).

---

## Phase 2 — Detection metadata

**Goal:** Build a single JSON manifest from D-Fire YOLO labels with absolute bounding boxes for FCOS (and optionally other detectors). Original images are not modified.

**Script:** `src/data/phase2_detection_metadata.py` (run as module).

```bash
python -m src.data.phase2_detection_metadata \
  --dataset-dir D-Fire \
  --output dfire_detection_annotations.json \
  --splits train test validation \
  --val-percent 10 \
  --resample-percent 25
```

**Options:**

- `--dataset-dir`: D-Fire root.
- `--output`: Path for the output JSON.
- `--splits`: Splits to include; validation can be derived from train as in Phase 1.
- `--fire-class-id` / `--smoke-class-id`: YOLO class ids for fire and smoke (default 1 and 0).
- `--resample-percent` / `--resample-seed`: Optional resampling with preserved class ratios.

**JSON shape:** Each entry under a split has `image_id`, `image_path`, `width`, `height`, `image_label_idx`, `image_label_name`, `num_annotations`, and `annotations` (list of objects with `class_idx`, `class_name`, `bbox_xyxy`, `bbox_yolo`). `bbox_xyxy` is in absolute pixels; `bbox_yolo` keeps normalized YOLO coords. `fcos/fcos_dataset.py` reads this JSON.

---

## CNN training (`main.py`)

- **Data:** Uses `src.data.load_data.retrieve_data_generator()` to build train/val/test `tf.data.Dataset`s from the Phase 1 CSVs (e.g. `processed_dfire/train_metadata.csv`).
- **Models:** Built from `src.models.cnn`; experiment configs live in `src.experiments.experiments`.
- **Logging:** Results are appended to `results/experiment_results.json` via `src.evaluation.evaluate.log_results()`.

Run all experiments:

```bash
python main.py
```

Run one experiment:

```bash
python main.py baseline
python main.py deeper_network
```

---

## FCOS detection training

FCOS uses **2 classes**: 0 = fire, 1 = smoke. The Phase 2 metadata uses 1-indexed labels (1=fire, 2=smoke); `src.data.phase2_detection_metadata` (and the FCOS dataset) provide `metadata_to_fcos_class()` / `fcos_to_metadata_class()` for conversion.

**Primary interface:** `fcos/fcos_training.ipynb` (local or Google Colab).

### Colab

1. Put the repo (or a copy) in Google Drive, e.g. `MyDrive/fire-detection`.
2. In the notebook, set `PROJECT_PATH` to that folder (e.g. `'/content/drive/MyDrive/fire-detection'`).
3. Ensure these are under the project path: `fcos/`, `src/` (including `src/data/phase2_detection_metadata.py`), `requirements.txt`, and the D-Fire dataset.
4. Runtime → Change runtime type → GPU, then run all cells. The notebook can generate the Phase 2 JSON if it’s missing.

### Local

1. Generate the detection JSON if needed:
   ```bash
   python -m src.data.phase2_detection_metadata --dataset-dir D-Fire --output dfire_detection_annotations.json --splits train test validation
   ```
2. Open the notebook from the repo root:
   ```bash
   jupyter notebook fcos/fcos_training.ipynb
   ```
3. Run all cells in order.

**FCOS files in `fcos/`:**

- `fcos.py`: FCOS model.
- `fcos_dataset.py`: PyTorch Dataset over the Phase 2 JSON; imports from `src.data.phase2_detection_metadata`.
- `fcos_training.ipynb`: Setup, training, checkpointing, TensorBoard, and visualizations.

Checkpoints and logs (e.g. `checkpoints/`, `logs/`, `training_history.json`) are created as configured in the notebook.
