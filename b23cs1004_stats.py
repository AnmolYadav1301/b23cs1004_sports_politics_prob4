import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# Logging Configuration

logging.basicConfig(
    filename="experiment_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Load dataset
dataset = pd.read_csv("dataset_title_filtered.csv")
texts = dataset["sentence"]
labels = dataset["label"]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


# Evaluation helper
def evaluate(clf, X_train_transformed, X_test_transformed, model_name):
    clf.fit(X_train_transformed, y_train)
    predictions = clf.predict(X_test_transformed)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    logging.info(
        f"{model_name} -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return accuracy, precision, recall, f1, predictions

# Part 1: Compare different feature representations (Naive Bayes)
feature_encodings = {
    "Bag_of_Words": CountVectorizer(),
    "TF-IDF": TfidfVectorizer(),
    "N-grams_1_2": CountVectorizer(ngram_range=(1,2))
}

feature_accuracies = {}

for feat_name, vectorizer in feature_encodings.items():
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb_model = MultinomialNB()
    acc, _, _, _, _ = evaluate(nb_model, X_train_vec, X_test_vec, feat_name)
    feature_accuracies[feat_name] = acc

# Plotting feature comparison
plt.figure(figsize=(8,5))
plt.bar(feature_accuracies.keys(), feature_accuracies.values(), color="skyblue")
plt.title("Feature Representation Comparison (Naive Bayes)")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("feature_comparison.png")
plt.close()

# Best performing feature
best_feature_name = max(feature_accuracies, key=feature_accuracies.get)
logging.info(f"Selected best feature encoding: {best_feature_name}")

# Part 2: Compare multiple ML models using the best feature
if best_feature_name == "Bag_of_Words":
    best_vectorizer = CountVectorizer()
elif best_feature_name == "TF-IDF":
    best_vectorizer = TfidfVectorizer()
else:
    best_vectorizer = CountVectorizer(ngram_range=(1,2))

X_train_vec = best_vectorizer.fit_transform(X_train)
X_test_vec = best_vectorizer.transform(X_test)

ml_models = {
    "Naive_Bayes": MultinomialNB(),
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Linear_SVM": LinearSVC(),
    "Random_Forest": RandomForestClassifier(n_estimators=100)
}

model_performance = {}
confusion_matrices = {}

for model_label, clf in ml_models.items():
    acc, prec, rec, f1, preds = evaluate(clf, X_train_vec, X_test_vec, model_label)
    model_performance[model_label] = acc
    confusion_matrices[model_label] = confusion_matrix(y_test, preds)

# Plotting model comparison
plt.figure(figsize=(8,5))
plt.bar(model_performance.keys(), model_performance.values(), color="coral")
plt.title(f"ML Model Comparison ({best_feature_name})")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

# Plot confusion matrices
for model_name, cm in confusion_matrices.items():
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_{model_name}.png")
    plt.close()

print("Experiment finished! Check experiment_log.txt and generated plots.")
