
# Spotify Genre Classification

This project uses Spotify audio features to predict the genre of a song. I built this as a machine learning classification project using data cleaning, train/test splitting, PCA, KMeans clustering, and supervised classification models.

## Project Overview

The goal was to predict one of 10 music genres using Spotify audio features such as danceability, energy, tempo, acousticness, loudness, speechiness, and popularity.

I started with a Logistic Regression baseline model, then added PCA and KMeans cluster features before training a stronger Random Forest classifier. I also tested a Linear SVM model for comparison.

## Dataset

The original dataset is not included in this repository because it was provided for course use. To run the project, place `musicData.csv` in the project folder or update the file path inside the script.

## Methods Used

- Data cleaning
- Missing value handling with median imputation
- One-hot encoding for categorical features
- Standardization of numeric features
- PCA dimensionality reduction
- KMeans clustering
- Logistic Regression baseline
- Random Forest classifier
- Linear SVM comparison
- Multi-class one-vs-rest AUC evaluation

## Final Results

| Model | Accuracy | Macro AUC |
|---|---:|---:|
| Baseline Logistic Regression | 0.5210 | 0.9024 |
| Random Forest with PCA + KMeans | 0.5406 | 0.9198 |
| Linear SVM with PCA + KMeans | 0.4990 | 0.8896 |

The final selected model was the Random Forest classifier with PCA and KMeans features. It achieved a final macro AUC of **0.9198**.

## Key Takeaways

The model performed best on genres that were more separated in the audio feature space, especially Classical. It struggled more with genres that overlap heavily, especially Rap and Hip-Hop. The feature importance results showed that popularity, PCA components, instrumentality, danceability, loudness, speechiness, and acousticness were important for prediction.

