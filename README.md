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
- ST-GCN was applied by adding of motion based features and attention layers whcih gave validation accuracy of 0.64 (highest so far).


## EMOKINE Dataset
The EmoKine Dataset is a specialized benchmark designed for emotion recognition from human body movements. Unlike most affective computing datasets that focus primarily on facial expressions or speech, EmoKine emphasizes the kinematic patterns of the full body, capturing how emotions are conveyed through posture, gesture, and motion. This makes it particularly relevant for computer vision systems that must operate in scenarios where facial cues are unreliable or unavailable.

EmoKine contains video recordings of participants performing acted emotional expressions across six universally recognized emotion categories: anger, sadness, joy, fear, disgust, and surprise. Each emotion is expressed through full-body actions rather than static poses, enabling the study of temporal motion cues. For each video, the dataset provides synchronized 2D keypoint sequences extracted using pose estimation tools such as MediaPipe or OpenPose. Some versions also include 3D pose representations, offering richer structural information for advanced temporal modeling.

Each recording includes metadata such as emotion label, participant ID, and trial number, enabling subject-independent experiments. The pose sequences typically consist of 17–33 body joints tracked across all frames, forming time-series data suitable for LSTMs, GRUs, Transformers, and Graph Convolutional Networks.

EmoKine is widely used in research on body-based emotion recognition, behavior understanding, and human–computer interaction, especially in settings like robotics or surveillance where facial visibility cannot be guaranteed. Its focus on movement dynamics makes it a valuable alternative to traditional facial-expression datasets, enabling the development of models that interpret emotion through body kinematics rather than appearance.

Overall, EmoKine provides a compact but highly expressive dataset that supports the study of how emotion is manifested through human motion, making it an important resource for kinematic analysis and multimodal affective computing.

## Future Work  
- Commit all the clean codes. 
- Finalize results and prepare final deliverables.
- Work on bonus parts of the deliverable.

