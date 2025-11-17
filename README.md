# Fire Detection with Machine Learning

This project implements a machine learning pipeline to detect fire and smoke in images. It uses a 3-class classification system (Fire, Smoke, Neutral) based on the D-Fire dataset.

The pipeline includes data preprocessing, feature extraction (color statistics, brightness, contrast), and baseline model training using KNN, Decision Trees, and Random Forests.

## Project Structure

To ensure the scripts run correctly, please organize your repository files as follows:

```
project_root/
├── Intro to ML Dataset/          # Place your original raw dataset here
├── phase1_classification_preprocess.py
├── models/                       # Folder containing python scripts
│   ├── train_baselines.py
│   ├── test_baselines.py
│   ├── utils.py
│   ├── saved_models/             # Stores .pkl files (created after training)
│   └── model_comparison_results.csv
└── processed_dfire/              # Generated automatically by preprocessing 
```

## Prerequisites

It is recommended to create a virtual environment and install the dependencies from the provided requirements.txt file.

```
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  

# On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run

### 1. Data Setup
Before running any scripts, you must add the original D-Fire dataset into the project.

1. Create a folder named "Intro to ML Dataset" in the root directory.
2. Place the original YOLO-format dataset (containing train/, test/, images/, labels/) inside this folder.

### 2. Generate Processed Data
We need to convert the raw YOLO images into a classification-friendly format (cropped square images sorted by class).

Run the preprocessing script from the root directory:

```
python phase1_classification_preprocess.py --dataset-dir "Intro to ML Dataset" --output-dir "processed_dfire"
```
- Input: Reads from "Intro to ML Dataset".
- Output: Creates a "processed_dfire" folder with cropped images organized by class (fire, smoke, nothing).

### 3. Training Models
Once the data is processed, you can train the baseline models (KNN, Decision Tree, Random Forest).

Navigate to the models directory and run the training script:
```
cd models
python train_baselines.py
```
- Process: This script loads images from ../processed_dfire, extracts features (RGB means, standard deviations, contrast, etc.), and trains the models.
- Output: Saves the trained models and scalers into the models/saved_models/ directory.

### 4. Testing Models
To evaluate the performance of the models, run the testing script:
```
cd models
python test_baselines.py
```
- Process: Loads the trained models and evaluates them on the test set.
- Output:
    - Prints classification reports (Precision, Recall, F1-Score).
    - Saves a summary CSV: model_comparison_results.csv.
    - Generates a confusion matrix plot: model_confusion_matrices.png.

## Testing Without Training (Pre-trained Models)

If you do not wish to retrain the models and want to use the pre-trained weights provided in this repository:

1. Ensure the "processed_dfire" folder exists (follow Step 2 above).
2. Ensure the "models/saved_models/" directory contains the .pkl files (e.g., knn.pkl, random_forest.pkl).
3. Simply run the test script:
```
cd models
python test_baselines.py
```
The script is designed to automatically look for the "saved_models" folder and load the existing models for evaluation.