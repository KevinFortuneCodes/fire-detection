# fire-detection

Fire detection software for Machine Learning class project.

## Phase 1 — Classification Preprocessing

**Goal:** Build a balanced classification dataset (fire / smoke / nothing) in an ImageFolder-compatible structure for training CNN and baseline models.

### Usage

```bash
python phase1_classification_preprocess.py \
  --dataset-dir D-Fire \
  --output-dir processed_dfire \
  --splits train validation test \
  --val-percent 10 \
  --resample-percent 30 \
  --log-every 25
```

### Key Options

- `--dataset-dir`: Path to D-Fire dataset root (`train/images`, `train/labels`, etc.)
- `--output-dir`: Directory where processed PNGs and metadata CSVs will be written
- `--splits`: List of splits to produce (validation is derived from training if not on disk)
- `--val-percent`: Fraction of training split to reserve for validation (default 10%)
- `--image-size`: Target square size for processed images (default 256)
- `--crop-strategy`: How to resize images - `center` (crop) or `stretch` (default: center)
- `--positive-class`: YOLO class ID treated as "fire" (default 1). Other labeled boxes become "smoke"; frames without labels become "nothing"
- `--resample-percent`: Optional down-sampling percentage (0-100) to reduce dataset size while preserving class ratios
- `--resample-seed`: Random seed for resampling (default 1234)
- `--skip-existing`: Skip images that already exist in output directory

### Output

- `processed_dfire/<split>/{fire,smoke,nothing}/<image>.png` - Processed square images organized by class
- `<output-dir>/<split>_metadata.csv` - Metadata CSV with image paths, labels, and bounding box counts

---

## Phase 2 — Detection Preprocessing

**Goal:** Convert YOLO label files into a JSON manifest with absolute bounding box coordinates for object detection training (e.g., FCOS). This keeps original images untouched and only generates metadata.

### Usage

```bash
python phase2_detection_metadata.py \
  --dataset-dir D-Fire \
  --output dfire_detection_annotations.json \
  --splits train test validation \
  --val-percent 10 \
  --resample-percent 30
```

### Key Options

- `--dataset-dir`: Root of the YOLO dataset
- `--output`: Path for the JSON manifest file
- `--splits`: List of splits to include (validation derived from training if not on disk)
- `--val-percent`: Percentage of resampled training split moved to validation when derived (default 10%)
- `--fire-class-id`: YOLO class ID for fire (default 1)
- `--smoke-class-id`: YOLO class ID for smoke (default 2)
- `--resample-percent`: Optional percentage (0-100) to sample images per split while preserving class ratios
- `--resample-seed`: Random seed for resampling (default 1234)

### Output Format

The JSON file contains entries with image metadata and bounding box annotations:

```json
{
  "image_id": "AoF07128",
  "image_path": ".../D-Fire/train/images/AoF07128.jpg",
  "width": 640,
  "height": 512,
  "image_label_idx": 1,
  "image_label_name": "fire",
  "annotations": [
    {
      "class_idx": 1,
      "class_name": "fire",
      "yolo_class_id": 1,
      "bbox_xyxy": [163.0, 397.0, 314.0, 487.0],
      "bbox_yolo": [0.2162, 0.7391, 0.1369, 0.1505]
    }
  ]
}
```

- `bbox_yolo`: Original normalized YOLO coordinates (x_center, y_center, width, height)
- `bbox_xyxy`: Absolute pixel coordinates (x1, y1, x2, y2) ready for detector training
- `image_label_idx`: Image-level label (1=fire, 2=smoke, 3=nothing, 4=fire_and_smoke)

---

## Phase 3 — FCOS Training

**Goal:** Train an FCOS-style object detector to detect fire and smoke bounding boxes.

### Important: Class Mapping

FCOS uses **2 classes** for detection:
- **Class 0**: Fire
- **Class 1**: Smoke

The metadata JSON uses class indices 1 (fire) and 2 (smoke), but the training code automatically maps these to FCOS classes (0=fire, 1=smoke) using helper functions.

### Training Setup

**The training notebook (`fcos/fcos_training.ipynb`) is the primary way to train the FCOS model.** It works both locally and on Google Colab with automatic setup.

#### Google Colab (Recommended)

1. **Upload to Google Drive**:
   - Create a folder (e.g., `fire-detection`)
   - Upload: `fcos/` folder, `phase2_detection_metadata.py`, `requirements.txt`, `D-Fire/` dataset

2. **Open notebook in Colab**:
   - Upload `fcos/fcos_training.ipynb` to Colab
   - Update `PROJECT_PATH` in Cell 1 to match your Google Drive folder path

3. **Enable GPU**:
   - Runtime → Change runtime type → GPU (T4 or better)

4. **Run all cells**:
   - Automatically mounts Drive, installs dependencies, generates metadata (if needed), and trains the model

#### Local Training

1. **Open the notebook**:
   ```bash
   jupyter notebook fcos/fcos_training.ipynb
   ```

2. **Run all cells in order**:
   - Automatically detects local environment
   - Generates metadata JSON if needed
   - Trains the model

### Training Configuration

Adjust parameters in the notebook's configuration cell:

- `batch_size`: Batch size (default 32)
- `num_epochs`: Number of training epochs (default 50)
- `learning_rate`: Learning rate (default 0.001)
- `target_size`: Input image size (default 800)
- `fpn_channels`: FPN channels (default 64)
- `stem_channels`: Stem channel sizes (default [64, 64])

### Monitoring Training

The notebook automatically:
- Saves checkpoints to `checkpoints/` (one per epoch + best model)
- Logs to TensorBoard in `logs/`
- Tracks training history in `checkpoints/training_history.json`

**View TensorBoard:**
```bash
tensorboard --logdir logs/
```

Or use the TensorBoard cell in the Colab notebook.

### Model Files

All FCOS-related files are in the `fcos/` folder:
- `fcos/fcos.py`: FCOS model implementation
- `fcos/fcos_dataset.py`: PyTorch Dataset class for loading detection data
- `fcos/fcos_training.ipynb`: Complete training notebook (works locally and on Colab)
- `fcos/__init__.py`: Package initialization file

### Model Output

The trained model outputs:
- `pred_boxes`: Tensor `(N, 4)` with bounding box coordinates (x1, y1, x2, y2)
- `pred_classes`: Tensor `(N,)` with class indices (0=fire, 1=smoke)
- `pred_scores`: Tensor `(N,)` with confidence scores

**Checkpoints saved:**
- `checkpoints/checkpoint_epoch_N.pth`: Checkpoint for each epoch
- `checkpoints/best_model.pth`: Best model based on validation loss
- `checkpoints/training_history.json`: Training history for plotting
