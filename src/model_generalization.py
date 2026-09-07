from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1: Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# Results
print("MODEL GENERALIZATION COMPARISON")
print("--------------------------------")
print(f"Decision Tree Accuracy : {dt_acc:.2f}")
print(f"Random Forest Accuracy : {rf_acc:.2f}")

if rf_acc > dt_acc:
    print("\nRandom Forest generalizes better.")
else:
    print("\nDecision Tree generalizes better.")
