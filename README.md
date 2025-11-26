# Emotion-Recognition-Body-and-Dance


## Overview  
This project builds an end to end pipeline for emotion recognition using human body movements and dance sequences.
Unlike traditional systems that rely on facial expressions or speech, this work focuses exclusively on movement dynamics, using pose estimation and machine learning to classify emotional states.


## Contributions on Kinematic Actors Dataset

### Dataset Preparation  
- BVH files were parsed taking into consideration the skleton structure and capturing the motion of all the joints.
- Movement based features were created along with statistical properties such as mean, std etc.
- Total 1407 samples were used and distribution on emotion classes came out to be balanced.

### Modelling  

- Machine Learning models acted as baseline and were further used to compute feature importance for the selection of top 50 features.
- Random Forest, SVM and Gradient Boosting models were able to achieve the score ranging from 0.51 to 0.55 
- MLP Classifier trained using parameters from optuna optimization on the top 50 engineered features was able to achieve 0.60 F1 score.
- It has been established that this dataset will require a lot of efforts to capture temporal variations per frame so going with cleaner version of EMOKINE dataset.  


## Future Work  
- LSTM and GCN sequence models on EMOKINE Dataset
