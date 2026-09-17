from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import os
import pandas as pd

# ==========================================
# 1. Create Results Folder
# ==========================================

os.makedirs("results", exist_ok=True)


# ==========================================
# 2. Load Dataset
# ==========================================

data = load_breast_cancer()

X = data.data
y = data.target


# ==========================================
# 3. Split Data into Training and Testing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 4. Model 1 - Decision Tree
# ==========================================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_train_prediction = dt.predict(X_train)
dt_test_prediction = dt.predict(X_test)

dt_train_acc = accuracy_score(
    y_train,
    dt_train_prediction
)

dt_acc = accuracy_score(
    y_test,
    dt_test_prediction
)


# ==========================================
# 5. Model 2 - Random Forest
# ==========================================

rf = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

rf.fit(X_train, y_train)

rf_train_prediction = rf.predict(X_train)
rf_test_prediction = rf.predict(X_test)

rf_train_acc = accuracy_score(
    y_train,
    rf_train_prediction
)

rf_acc = accuracy_score(
    y_test,
    rf_test_prediction
)


# ==========================================
# 6. Generalization Gap
# ==========================================

dt_gap = dt_train_acc - dt_acc
rf_gap = rf_train_acc - rf_acc


# ==========================================
# 7. Evaluation Metrics
# ==========================================

dt_precision = precision_score(
    y_test,
    dt_test_prediction
)

dt_recall = recall_score(
    y_test,
    dt_test_prediction
)

dt_f1 = f1_score(
    y_test,
    dt_test_prediction
)


rf_precision = precision_score(
    y_test,
    rf_test_prediction
)

rf_recall = recall_score(
    y_test,
    rf_test_prediction
)

rf_f1 = f1_score(
    y_test,
    rf_test_prediction
)


# ==========================================
# 8. Display Results
# ==========================================

print("\n======================================")
print("MODEL GENERALIZATION COMPARISON")
print("======================================")


print("\nDecision Tree")
print("----------------------------")

print(f"Precision         : {dt_precision:.2f}")
print(f"Recall            : {dt_recall:.2f}")
print(f"F1 Score          : {dt_f1:.2f}")
print(f"Training Accuracy : {dt_train_acc:.2f}")
print(f"Testing Accuracy  : {dt_acc:.2f}")
print(f"Generalization Gap: {dt_gap:.2f}")


print("\nRandom Forest")
print("----------------------------")

print(f"Precision         : {rf_precision:.2f}")
print(f"Recall            : {rf_recall:.2f}")
print(f"F1 Score          : {rf_f1:.2f}")
print(f"Training Accuracy : {rf_train_acc:.2f}")
print(f"Testing Accuracy  : {rf_acc:.2f}")
print(f"Generalization Gap: {rf_gap:.2f}")


# ==========================================
# 9. Final Comparison
# ==========================================

if rf_acc > dt_acc:

    print("\nRandom Forest generalizes better.")

elif dt_acc > rf_acc:

    print("\nDecision Tree generalizes better.")

else:

    print("\nBoth models have the same test accuracy.")


# ==========================================
# 10. Accuracy Comparison Graph
# ==========================================

models = [
    "Decision Tree",
    "Random Forest"
]

training_accuracy = [
    dt_train_acc,
    rf_train_acc
]

testing_accuracy = [
    dt_acc,
    rf_acc
]

x = range(len(models))


plt.figure(figsize=(8, 5))


plt.bar(
    [i - 0.2 for i in x],
    training_accuracy,
    width=0.4,
    label="Training Accuracy"
)


plt.bar(
    [i + 0.2 for i in x],
    testing_accuracy,
    width=0.4,
    label="Testing Accuracy"
)


plt.xticks(
    list(x),
    models
)

plt.ylabel("Accuracy")

plt.xlabel("Models")

plt.title(
    "Training vs Testing Accuracy"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "results/accuracy_comparison.png"
)

plt.close()


# ==========================================
# 11. Decision Tree Confusion Matrix
# ==========================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    dt_test_prediction,
    display_labels=data.target_names
)

plt.title(
    "Decision Tree Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    "results/decision_tree_confusion_matrix.png"
)

plt.close()


# ==========================================
# 12. Random Forest Confusion Matrix
# ==========================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_test_prediction,
    display_labels=data.target_names
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.tight_layout()


plt.savefig(
    "results/random_forest_confusion_matrix.png"
)

plt.close()


# ==========================================
# 13. Final Message
# ==========================================

print("\n======================================")
print("RESULT FILES CREATED")
print("======================================")

print("1. accuracy_comparison.png")
print("2. decision_tree_confusion_matrix.png")
print("3. random_forest_confusion_matrix.png")
# ==========================================
# 14. Save Final Results Table
# ==========================================

results = {
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],

    "Training Accuracy": [
        dt_train_acc,
        rf_train_acc
    ],

    "Testing Accuracy": [
        dt_acc,
        rf_acc
    ],

    "Generalization Gap": [
        dt_gap,
        rf_gap
    ],

    "Precision": [
        dt_precision,
        rf_precision
    ],

    "Recall": [
        dt_recall,
        rf_recall
    ],

    "F1 Score": [
        dt_f1,
        rf_f1
    ]
}

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nFinal results saved to:")
print("results/model_comparison.csv")