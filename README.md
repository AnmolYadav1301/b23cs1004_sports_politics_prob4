# b23cs1004_sports_politics_prob4
# Sports vs Politics Text Classification

## Project Overview

This project implements a supervised machine learning classifier that categorizes text sentences into two domains:

- **Sports**
- **Politics**

The objectives are to:

1. Build a large-scale labeled dataset automatically.
2. Compare different feature representation techniques.
3. Evaluate multiple machine learning models.
4. Perform quantitative comparison using standard metrics.
5. Deploy the best-performing model for interactive classification.

---

## Problem Statement

The goal is to design a system that reads a text sentence or document and predicts whether it belongs to:

- **Sports**
- **Politics**

The system compares three types of feature representations:

- Bag of Words (BoW)
- TF-IDF
- N-grams (Unigrams + Bigrams)

It also evaluates four machine learning models:

- Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest

---

## Dataset Construction

### Data Source

The dataset is generated automatically from Wikipedia category pages:

- [Sports Category](https://en.wikipedia.org/wiki/Category:Sports)
- [Politics Category](https://en.wikipedia.org/wiki/Category:Politics)

### Keyword Extraction

- Domain-specific keywords were extracted from article titles (up to 500 per domain).  
- Only relevant titles containing keywords were used.  
- Disambiguation and meta pages were excluded.  

### Sentence Extraction

- Paragraphs from valid pages were split into sentences using NLTK.  
- Sentences shorter than 8 words were discarded.  
- Final dataset contains **50,000 sentences** (25,000 per class).  

---

## Features & Models

### Feature Representations

1. **Bag of Words (BoW)** – Word frequency counts.  
2. **TF-IDF** – Highlights discriminative words.  
3. **N-grams (1,2)** – Captures contextual word sequences.

### Machine Learning Models

1. **Naive Bayes**  
2. **Logistic Regression**  
3. **Linear SVM**  
4. **Random Forest**  

Evaluation metrics used:

- Accuracy  
- Precision  
- Recall  
- F1-Score  

---

## How to Run

### Step 1: Generate Dataset

Run the dataset generation script:

```bash
python3 generate_dataset.py

This will crawl Wikipedia, filter relevant articles, extract sentences, and save the dataset as:

dataset_title_filtered.csv

### Step 2: Train & Use Classifier

Run the classifier script:


```bash
python3 b23cs1004_classifier.py


The script will train the best-performing model if not already saved.

Once loaded, you can type any sentence and the model will predict:

SPORTS 🏆
POLITICS 🏛

Type exit to quit the interactive mode.