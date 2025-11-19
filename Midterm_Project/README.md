# Galaxy Morphology Classification

## Description
This project builds an image classifier for galaxy morphology using the Galaxy Zoo dataset (Kaggle: Galaxy Zoo - The Galaxy Challenge).  
We simplify the original task to a coarse **3-class** classification: `elliptical`, `spiral`, `other` (mergers/artifacts/uncertain).

The pipeline includes:
- data loading & cleaning
- EDA and feature analysis
- training a CNN with transfer learning (EfficientNetB0)
- model evaluation and visualizations (confusion matrix + Grad-CAM)
- a saved model and a small Flask service for predictions


## Data
- **Source**: Kaggle competition `galaxy-zoo-the-galaxy-challenge`.
- **How to obtain:**  
  1. Create a Kaggle account and retrieve the dataset from: `https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data`  
  2. Download `images_training_rev1.zip` and `training_solutions_rev1.zip`.  
  3. Unzip into `data/images_training/` and `data/labels/` respectively.  
  Alternatively, run:
  ```bash
  # requires kaggle CLI configured with ~/.kaggle/kaggle.json
  kaggle competitions download -c galaxy-zoo-the-galaxy-challenge -p data
  unzip data/images_training_rev1.zip -d data/images_training
  unzip data/training_solutions_rev1.zip -d data/labels
