# Fire and Smoke Detection with Machine Learning

This project implements a comprehensive machine learning pipeline to detect fire, smoke, and their co-occurrence in images. Using the D-Fire dataset, we explore both simple (10 features) and enhanced (26 features) feature extraction methods to train and evaluate baseline models including KNN, Decision Trees, and Random Forests, with and without hyperparameter tuning.

## Project Structure

```bash
project_root/
├── Intro to ML Dataset/          # Original YOLO-format dataset (train/, test/, images/, labels/)
├── phase1_classification_preprocess.py
├── phase2_detection_metadata.py
├── models/                       # ML model scripts
│   ├── train_baselines.py        # Training script
│   ├── test_baselines.py         # Testing/evaluation script
│   ├── utils.py                  # Feature extraction and utilities
│   ├── saved_models/             # Stores trained models (created after training)
│   ├── model_reports/            # Detailed performance summaries
│   └── confusion_matrices/       # Confusion matrix visualizations
└── processed_dfire/              # Preprocessed images (created by preprocessing)
```

## Prerequisites

Create a virtual environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run

### 1. Data Setup
Before running any scripts, you must add the original D-Fire dataset:

1. Create a folder named `"Intro to ML Dataset"` in the root directory
2. Place the original YOLO-format dataset (containing train/, test/, images/, labels/) inside this folder

### 2. Generate Processed Data
Convert raw YOLO images into a classification-friendly format (4 classes: fire, smoke, nothing, fire_and_smoke):

```bash
python phase1_classification_preprocess.py --dataset-dir "Intro to ML Dataset" --output-dir "processed_dfire"
```

**Output**: Creates a `"processed_dfire"` folder with cropped images organized by class.

### 3. Training Models
Train baseline models with different configurations:

```bash
cd models

# Train with enhanced features (default)
python train_baselines.py --feature-type enhanced

# Train with simple features
python train_baselines.py --feature-type simple

# Skip hyperparameter tuning
python train_baselines.py --feature-type enhanced --no-tuning

# Adjust sample size (default: 17221)
python train_baselines.py --max-samples 4000
```

**Output**: 
- Trained models saved in `models/saved_models/` as `.pkl` files
- Multiple configurations: `[model]_[feature_type].pkl` and `[model]_[feature_type]_tuned.pkl`

### 4. Testing Models
Evaluate trained models on the test set:

```bash
cd models

# Test enhanced models (default)
python test_baselines.py --feature-type enhanced

# Test simple models
python test_baselines.py --feature-type simple

# Test tuned models
python test_baselines.py --feature-type enhanced --tuned

# Test all configurations and generate comparison
python test_baselines.py --feature-type enhanced
python test_baselines.py --feature-type simple
python test_baselines.py --feature-type enhanced --tuned
python test_baselines.py --feature-type simple --tuned
```

**Output**:
- Console: Classification reports, accuracy, precision, recall, F1-scores
- `model_reports/`: Detailed text summaries of all metrics
- `confusion_matrices/`: PNG visualizations and CSV data for confusion matrices
- Console: Performance comparison table across all tested models

## Important Notes

### No Pre-trained Models Included
**This repository does NOT include pre-trained model files.** You must train the models using `train_baselines.py` before testing. The `saved_models/` directory will be created automatically during training and will contain the `.pkl` files needed for testing.

### Model Configurations
The pipeline supports four configurations for each model type:
1. **Simple Features (10 features)**: Basic RGB statistics, brightness, contrast, and color ratios
2. **Enhanced Features (26 features)**: Extended features including spatial analysis, edge detection, and color uniformity
3. **Default Parameters**: Standard sklearn parameters
4. **Tuned Parameters**: Models with hyperparameter optimization (when `--tuned` flag is used)

### Expected Results
Based on our experiments, the best performing model is **Random Forest with Enhanced Features and hyperparameter tuning**, achieving approximately **64.8% accuracy**. However, all baseline models struggle with distinguishing between smoke and neutral backgrounds, and particularly with the `fire_and_smoke` class, highlighting the limitations of handcrafted features and the need for deep learning approaches.
