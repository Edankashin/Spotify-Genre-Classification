#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
random.seed(15906048)

import pandas as pd
import numpy as np


df = pd.read_csv("musicData.csv")

# Characteristic information
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nGenre counts:")
print(df["music_genre"].value_counts(dropna=False))



df = df.dropna(how="all")
df = df.dropna(subset=["music_genre"])
df.columns = df.columns.str.strip()
text_columns = ["instance_id", "artist_name", "track_name", "key", "mode", "obtained_date", "music_genre"]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Convert tempo to numeric
df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")

# Check the cleaned dataset
print("Shape after basic cleaning:", df.shape)

print("\nGenre counts after cleaning:")
print(df["music_genre"].value_counts())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nData types after cleaning:")
print(df.dtypes)

#train/test split

SEED = 15906048


# Make sure the random seed is set
import random
random.seed(SEED)
np.random.seed(SEED)

# Random 500 songs from each genre
test_df = (
    df.groupby("music_genre", group_keys=False)
      .sample(n=500, random_state=SEED)
)

# Everything not in the test set becomes training data
train_df = df.drop(test_df.index)

X_train = train_df.drop(columns=["music_genre"])
y_train = train_df["music_genre"]

X_test = test_df.drop(columns=["music_genre"])
y_test = test_df["music_genre"]

# Check the sizes
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTraining genre counts:")
print(y_train.value_counts())

print("\nTest genre counts:")
print(y_test.value_counts())

# Check for leakage
overlap = set(train_df.index).intersection(set(test_df.index))
print("\nNumber of overlapping rows between train and test:", len(overlap))



#Baseline model

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

#feautures
numeric_features = [
    "danceability",
    "duration_ms",
    "energy",
    "tempo",
    "valence",
    "popularity",
    "acousticness",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
]

categorical_features = [
    "key",
    "mode"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


baseline_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
y_proba = baseline_model.predict_proba(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)

# AUC
auc = roc_auc_score(
    y_test,
    y_proba,
    multi_class="ovr",
    average="macro",
    labels=baseline_model.classes_
)

print("Baseline Accuracy:", accuracy)
print("Baseline Macro AUC:", auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# PCA scree plot

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Only using numeric/audio features for PCA
pca_numeric_features = numeric_features.copy()

pca_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_numeric_clean = pca_preprocessor.fit_transform(X_train[pca_numeric_features])
X_test_numeric_clean = pca_preprocessor.transform(X_test[pca_numeric_features])

pca_full = PCA()
X_train_pca_full = pca_full.fit_transform(X_train_numeric_clean)

eigenvalues = pca_full.explained_variance_


explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("Eigenvalues:")
for i, value in enumerate(eigenvalues, start=1):
    print(f"PC{i}: {value:.4f}")

print("\nExplained variance ratio:")
for i, value in enumerate(explained_variance_ratio, start=1):
    print(f"PC{i}: {value:.4f}")

print("\nCumulative explained variance:")
for i, value in enumerate(cumulative_variance, start=1):
    print(f"First {i} PCs: {value:.4f}")

# Scree plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues)
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.title("PCA Scree Plot")
plt.xticks(range(1, len(eigenvalues) + 1))
plt.grid(axis="y")
plt.savefig("pca_scree_plot.png", dpi=300)
plt.show()

#PCA-transformed train and test data

N_PCA_COMPONENTS = 6

pca = PCA(n_components=N_PCA_COMPONENTS, random_state=SEED)

X_train_pca = pca.fit_transform(X_train_numeric_clean)
X_test_pca = pca.transform(X_test_numeric_clean)

print("X_train_pca shape:", X_train_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)
print("Total variance explained by selected PCs:", pca.explained_variance_ratio_.sum())

# STEP 5C: KMeans elbow and silhouette analysis

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

cluster_range = range(2, 16)

inertia_scores = []
silhouette_scores = []

for k in cluster_range:
    kmeans_temp = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=10
    )
    
    cluster_labels = kmeans_temp.fit_predict(X_train_pca)
    
    inertia_scores.append(kmeans_temp.inertia_)
    
    sil_score = silhouette_score(
        X_train_pca,
        cluster_labels,
        sample_size=5000,
        random_state=SEED
    )
    
    silhouette_scores.append(sil_score)
    
    print(f"k={k}: inertia={kmeans_temp.inertia_:.2f}, silhouette={sil_score:.4f}")

# Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Distortion / Inertia")
plt.title("KMeans Elbow Method")
plt.grid(True)
plt.savefig("kmeans_elbow_plot.png", dpi=300)
plt.show()

# Silhouette plot
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Average Silhouette Score")
plt.title("KMeans Silhouette Scores")
plt.grid(True)
plt.savefig("kmeans_silhouette_plot.png", dpi=300)
plt.show()



#Final PCA and KMeans feature creation

N_CLUSTERS = 6

final_kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=SEED,
    n_init=10
)

train_cluster_labels = final_kmeans.fit_predict(X_train_pca)
test_cluster_labels = final_kmeans.predict(X_test_pca)

# Make copies of the original train/test feature sets
X_train_final = X_train.copy()
X_test_final = X_test.copy()

for i in range(N_PCA_COMPONENTS):
    X_train_final[f"PC{i+1}"] = X_train_pca[:, i]
    X_test_final[f"PC{i+1}"] = X_test_pca[:, i]


X_train_final["kmeans_cluster"] = train_cluster_labels.astype(str)
X_test_final["kmeans_cluster"] = test_cluster_labels.astype(str)

print("Final training feature shape:", X_train_final.shape)
print("Final test feature shape:", X_test_final.shape)

print("\nCluster counts in training data:")
print(pd.Series(train_cluster_labels).value_counts().sort_index())

print("\nCluster counts in test data:")
print(pd.Series(test_cluster_labels).value_counts().sort_index())





#Train final Random Forest classifier

from sklearn.ensemble import RandomForestClassifier

pca_feature_names = [f"PC{i+1}" for i in range(N_PCA_COMPONENTS)]

final_numeric_features = numeric_features + pca_feature_names
final_categorical_features = categorical_features + ["kmeans_cluster"]

final_numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

final_categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

final_preprocessor = ColumnTransformer(
    transformers=[
        ("num", final_numeric_transformer, final_numeric_features),
        ("cat", final_categorical_transformer, final_categorical_features)
    ]
)

final_model = Pipeline(steps=[
    ("preprocessor", final_preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1
    ))
])

final_model.fit(X_train_final, y_train)

final_y_pred = final_model.predict(X_test_final)
final_y_proba = final_model.predict_proba(X_test_final)

final_accuracy = accuracy_score(y_test, final_y_pred)

final_auc = roc_auc_score(
    y_test,
    final_y_proba,
    multi_class="ovr",
    average="macro",
    labels=final_model.classes_
)

print("Final Model Accuracy:", final_accuracy)
print("Final Model Macro AUC:", final_auc)

print("\nFinal Model Classification Report:")
print(classification_report(y_test, final_y_pred))

#Trying SVM model for comparison

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

svm_model = Pipeline(steps=[
    ("preprocessor", clone(final_preprocessor)),
    ("classifier", CalibratedClassifierCV(
        estimator=LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=SEED
        ),
        cv=3
    ))
])

svm_model.fit(X_train_final, y_train)

svm_y_pred = svm_model.predict(X_test_final)
svm_y_proba = svm_model.predict_proba(X_test_final)

svm_accuracy = accuracy_score(y_test, svm_y_pred)

svm_auc = roc_auc_score(
    y_test,
    svm_y_proba,
    multi_class="ovr",
    average="macro",
    labels=svm_model.classes_
)

print("SVM Accuracy:", svm_accuracy)
print("SVM Macro AUC:", svm_auc)

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_y_pred))


#Compare model results

model_comparison = pd.DataFrame({
    "Model": [
        "Baseline Logistic Regression",
        "Random Forest with PCA + KMeans",
        "Linear SVM with PCA + KMeans"
    ],
    "Accuracy": [
        accuracy,
        final_accuracy,
        svm_accuracy
    ],
    "Macro AUC": [
        auc,
        final_auc,
        svm_auc
    ]
})

print(model_comparison)


#ROC curves for final model

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc as sklearn_auc


class_names = final_model.classes_

y_test_binarized = label_binarize(y_test, classes=class_names)

# Plot one ROC curve per genre
plt.figure(figsize=(9, 7))

for i, genre in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], final_y_proba[:, i])
    genre_auc = sklearn_auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{genre} AUC = {genre_auc:.3f}")

# Random guessing line
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves for Final Model")
plt.legend(loc="lower right", fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("final_model_roc_curve.png", dpi=300)
plt.show()


#Visualize genres in PCA space

pca_plot_df = pd.DataFrame({
    "PC1": X_test_pca[:, 0],
    "PC2": X_test_pca[:, 1],
    "Genre": y_test.values,
    "KMeans Cluster": test_cluster_labels.astype(str)
})

plt.figure(figsize=(9, 7))

for genre in sorted(pca_plot_df["Genre"].unique()):
    genre_data = pca_plot_df[pca_plot_df["Genre"] == genre]
    plt.scatter(
        genre_data["PC1"],
        genre_data["PC2"],
        s=12,
        alpha=0.6,
        label=genre
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Genres Visualized in PCA Space")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("genre_pca_visualization.png", dpi=300)
plt.show()

#Visualize KMeans clusters in PCA space

plt.figure(figsize=(9, 7))

for cluster in sorted(pca_plot_df["KMeans Cluster"].unique()):
    cluster_data = pca_plot_df[pca_plot_df["KMeans Cluster"] == cluster]
    plt.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        s=12,
        alpha=0.6,
        label=f"Cluster {cluster}"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("KMeans Clusters Visualized in PCA Space")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_pca_visualization.png", dpi=300)
plt.show()


# Extra visualization: normalized confusion matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_names = final_model.classes_

cm = confusion_matrix(
    y_test,
    final_y_pred,
    labels=class_names,
    normalize="true"
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format=".2f"
)

plt.title("Normalized Confusion Matrix for Final Model")
plt.tight_layout()
plt.savefig("final_model_confusion_matrix.png", dpi=300)
plt.show()

# Extra analysis: printing the most common genre confusions

cm_counts = confusion_matrix(
    y_test,
    final_y_pred,
    labels=class_names
)

confusion_pairs = []

for actual_index, actual_genre in enumerate(class_names):
    for predicted_index, predicted_genre in enumerate(class_names):
        if actual_index != predicted_index:
            confusion_pairs.append({
                "Actual Genre": actual_genre,
                "Predicted Genre": predicted_genre,
                "Count": cm_counts[actual_index, predicted_index],
                "Percent of Actual Genre": cm[actual_index, predicted_index]
            })

confusion_pairs_df = pd.DataFrame(confusion_pairs)

confusion_pairs_df = confusion_pairs_df.sort_values(
    by="Count",
    ascending=False
)

print("Top 15 genre confusions:")
print(confusion_pairs_df.head(15))



# Extra analysis: genre makeup of each KMeans cluster

cluster_genre_counts = pd.crosstab(
    pca_plot_df["KMeans Cluster"],
    pca_plot_df["Genre"]
)

cluster_genre_percentages = pd.crosstab(
    pca_plot_df["KMeans Cluster"],
    pca_plot_df["Genre"],
    normalize="index"
) * 100

print("Cluster by genre counts:")
print(cluster_genre_counts)

print("\nCluster by genre percentages:")
print(cluster_genre_percentages.round(2))


# Extra analysis: feature importance from Random Forest

feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
feature_importances = final_model.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("Top 20 most important features:")
print(importance_df.head(20))