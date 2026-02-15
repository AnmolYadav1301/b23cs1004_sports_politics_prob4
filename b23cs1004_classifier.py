import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# File paths
# -----------------------------
MODEL_PATH = "sports_politics_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
DATA_PATH = "dataset_title_filtered.csv"

# -----------------------------
# Train & persist the model
# -----------------------------
def build_and_store_model():
    print("Starting model training...")

    data = pd.read_csv(DATA_PATH)
    sentences = data["sentence"]
    labels = data["label"]

    # Using 1-2 grams for better context
    vec = CountVectorizer(ngram_range=(1,2))
    X_vec = vec.fit_transform(sentences)

    clf = MultinomialNB()
    clf.fit(X_vec, labels)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vec, VECTORIZER_PATH)

    print("Training complete. Model and vectorizer saved.\n")


# -----------------------------
# Load persisted model & vectorizer
# -----------------------------
def load_saved_model():
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return classifier, vectorizer


# -----------------------------
# Interactive prediction loop
# -----------------------------
def interactive_classification():
    # Check if model exists, otherwise train
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        build_and_store_model()

    classifier, vectorizer = load_saved_model()
    print("Classifier is ready! Type a sentence or 'exit' to quit.\n")

    while True:
        text = input("Input: ").strip()
        if text.lower() == "exit":
            print("Goodbye!")
            break

        text_vec = vectorizer.transform([text])
        pred = classifier.predict(text_vec)[0]
        prob = classifier.predict_proba(text_vec)[0]

        if pred == 0:
            category = "SPORTS 🏆"
            confidence = prob[0]
        else:
            category = "POLITICS 🏛"
            confidence = prob[1]

        print(f"Prediction: {category}")
        print(f"Confidence: {confidence:.4f}\n")


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    interactive_classification()
