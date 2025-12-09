# Emotion Recognition in Body and Dance

## Overview
This project builds an end-to-end system for recognizing human emotions from body movement and dance sequences, without using facial expressions or speech. The goal is to understand emotion purely from posture, motion dynamics, and full-body kinematics. The system uses pose-based modeling, feature engineering, and deep learning (LSTM & ST-GCN) to classify emotions such as Angry, Disgust, Fearful, Happy, Neutral, Sad, and Surprise.

This repository contains all code notebooks, data processing scripts, models, and evaluation results developed for the project.

## Repository Structure & File Descriptions
**1. Kinematic_Dataset_and_Modelling.ipynb**

Preprocessing pipeline for the Kinematic Actors Dataset:

* Custom BVH parser to extract skeletal hierarchy & motion channels
* Extraction of 3D joint trajectories (T × V × 3)
* Motion-based feature engineering (velocity, acceleration, movement intensity)
* Dataset construction with 1400+ samples
* Creation of CSV-based engineered features

**2. LSTM_ON_KINEMATICS.ipynb**

Implements the full Motion-LSTM model:

* Addition of motion features (position + velocity + acceleration)
* Data augmentation (noise, Mixup)
* Channel attention + bidirectional LSTM
* Temporal attention module
* Training loop with scheduler, gradient clipping, and checkpoint saving

Evaluation: Confusion matrix, per-class accuracy, probability inspection, 5-fold cross-validation
(Methods and evaluation summarized from report pages 6–13)

**3. STGCN_Kinematic_Dataset.ipynb**

Implementation of Spatio-Temporal Graph Convolutional Network (ST-GCN):

* Skeleton graph construction
* Motion normalization & resampling
* Multi-attention ST-GCN architecture (temporal, spatial, channel attention)
* Training + K-fold cross-validation
* Per-class precision, recall, and F1 visualizations
(Based on ST-GCN discussion pp. 5–10)

**4. MLP_Experiments_(Kinematic_Dataset).ipynb**

Feature-based classification using engineered features:

* Mutual Information–based feature selection
* Optuna hyperparameter optimization
* MLP classifier achieving ≈60% accuracy

**5. 640_project_dataset_building_and_testing_on_l...ipynb**

Initial modeling tests, early data inspection, and prototype experiments (first modeling stage).

**6. Final_Project_640.ipynb**

Covers EMOKINE dataset experiments:

* 2D & 3D pose extraction
* LSTM and Transformer tests
* Classical ML with engineered features
* Leave-One-Out Cross-Validation results (~3.7% generalization)

**7. features_dataset.csv**

CSV file containing engineered kinematic features (400–500 features per motion trial) used for ML baselines and MLP.

**8. results/ Folder**

Contains all saved outputs:

* Confusion matrices
* Cross-validation summaries
* ST-GCN fold-wise results
* Visualization plots


## Dataset Details

**Kinematic Actors Dataset (Primary Dataset)**
1,402 BVH motion-capture trials labeled with 7 emotions.
BVH files contain:

HIERARCHY: skeletal structure (58 joints + root + end sites)

MOTION: frame-wise joint rotations & root translation

This dataset provides rich 3D kinematic signals suitable for LSTM and ST-GCN models.

**EMOKINE Dataset (Initial Exploration)**
Used for early experiments; ultimately abandoned due to:

* Very small sample size
* Single performer → low variability
* Subtle emotional differences
* Deep models failed LOOCV (~3.7%)

**Modeling Approaches**
Baseline Models:

* SVM, Random Forest, Gradient Boosting -> Achieved 0.52–0.55 accuracy
  Useful for feature importance & establishing benchmarks

* MLP Classifier (Engineered Features) -> 150 MI-selected features
  Optuna-optimized architecture -> Achieved 0.60 accuracy with low variance

* LSTM (Motion-LSTM)
 Input: 3D joints + motion features
 Channel + temporal attention
 Motion augmentation (noise, Mixup)
 Final accuracy: 0.593
 Strongest classes: Neutral, Happy

* ST-GCN
  Graph-based model leveraging skeletal structure
  Integrated multi-attention design
  Cross-validation accuracy: 0.567
  Strong for Neutral, Fearful, Happy

**Evaluation Summary**
Across all experiments:

 1. Neutral is the most consistently recognized emotion
 2. Surprise and Disgust are the hardest for models
 3. LSTM provides best sequence-level understanding
 4. ST-GCN captures spatial structure better than MLP and baselines
 5. MLP offers strong performance with engineered features
 6. EMOKINE results revealed dataset limitations → motivated switch
 7. K-fold and confusion-matrix evaluations confirm robustness and class-level behavior.

**Setup Instructions**
1. Clone the Repository : git clone https://github.com/<your-username>/Emotion-Recognition-Body-and-Dance.git
cd Emotion-Recognition-Body-and-Dance

2. Install Dependencies
   pip install -r requirements.txt

3. Dataset Setup
  Place BVH files from the Kinematic Actors Dataset into the folder:
  /data/bvh_files/

4. Run Preprocessing

Use: Kinematic_Dataset_and_Modelling.ipynb
to generate the cleaned dataset and engineered features.

5. Train Models
   * LSTM: LSTM_ON_KINEMATICS.ipynb

   * ST-GCN: STGCN_Kinematic_Dataset.ipynb

   * MLP: MLP_Experiments_(Kinematic_Dataset).ipynb

   * View Results
     All confusion matrices, metric summaries, and plots are stored in: /results/

## Future Work

1. Add multimodal integration (movement + audio or face)
2. Improve augmentation strategies
3. Build a web-based demo interface
4. Extend dataset with more performers and naturalistic motion
