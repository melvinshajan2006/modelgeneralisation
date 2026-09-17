 # Model Generalization

## 1. Project Title

Model Generalization Using Decision Tree and Random Forest

---

## 2. Objective

The main objective of this project is to compare two machine learning
models and study how well they perform on unseen data.

The models used are:

1. Decision Tree
2. Random Forest

---

## 3. Dataset

The project uses the Breast Cancer Wisconsin dataset available from
Scikit-learn.

The dataset contains:

- 569 samples
- 30 features
- 2 target classes

The target classes are:

- Malignant
- Benign

---

## 4. Methodology

The project follows these steps:

1. Load the dataset.
2. Separate the input features and target.
3. Divide the data into training and testing data.
4. Train a Decision Tree model.
5. Train a Random Forest model.
6. Calculate training accuracy.
7. Calculate testing accuracy.
8. Calculate the generalization gap.
9. Calculate Precision, Recall and F1 Score.
10. Compare the two models.

---

## 5. Train-Test Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

The training data is used to teach the models.

The testing data is used to evaluate their performance on unseen data.

---

## 6. Models Used

### Decision Tree

A Decision Tree makes predictions by creating a sequence of
feature-based decisions.

### Random Forest

Random Forest combines multiple decision trees to make predictions.

In this project, the Random Forest uses 50 trees.

---

## 7. Evaluation Metrics

The following metrics are used:

### Accuracy

Measures the percentage of correct predictions.

### Precision

Measures how many predicted positive results are actually positive.

### Recall

Measures how many actual positive cases are correctly identified.

### F1 Score

Combines Precision and Recall into one measurement.

### Generalization Gap

Generalization Gap is calculated as:

Training Accuracy - Testing Accuracy

---

## 8. Results

| Model | Training Accuracy | Testing Accuracy | Generalization Gap | Precision | Recall | F1 Score |
|------|-------------------|------------------|--------------------|-----------|--------|----------|
| Decision Tree | 1.00 | 0.9474 | 0.0526 | 0.96 | 0.96 | 0.96 |
| Random Forest | 1.00 | 0.9649 | 0.0351 | 0.96 | 0.99 | 0.97 |

---

## 9. Result Analysis

Both models achieved 100% training accuracy.

On the test data:

- Decision Tree achieved approximately 94.74% accuracy.
- Random Forest achieved approximately 96.49% accuracy.

The Random Forest produced the higher measured test accuracy in this
experiment.

The Random Forest also had a smaller generalization gap.

---

## 10. Project Results

The project generates the following files:

- accuracy_comparison.png
- decision_tree_confusion_matrix.png
- random_forest_confusion_matrix.png
- model_comparison.csv

These files are stored in the `results` folder.

---

## 11. Conclusion

This project demonstrates the importance of testing machine learning
models on unseen data.

Training accuracy alone is not enough to understand model performance.

By comparing training and testing performance, the generalization
ability of different models can be studied.

In this experiment, Random Forest achieved higher test accuracy than
Decision Tree.

---

## 12. Technologies Used

- Python
- Scikit-learn
- Pandas
- Matplotlib
- VS Code
- GitHub
