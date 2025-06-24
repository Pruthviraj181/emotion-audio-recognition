# Emotion Audio Recognition with XGBoost

## Project Description
This project uses XGBoost classifier to recognize emotions from audio (speech) using MFCC features.

## Preprocessing
- Audio loaded using librosa
- MFCC (Mel-frequency cepstral coefficients) extracted
- Train/test split done

## Model Pipeline
- XGBoost Classifier trained on MFCC features
- Evaluated with accuracy, F1 score, confusion matrix

## Results
Accuracy: ~78%  
F1 Score: as per report in notebook

## How to run test script:

```bash
python src/test_model.py

