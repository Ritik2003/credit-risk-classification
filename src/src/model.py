import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report

df = pd.read_csv("data/credit_data.csv")

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(class_weight="balanced", n_estimators=300)
model.fit(X_train, y_train)

preds = model.predict(X_test)

recall = recall_score(y_test, preds)
print("Recall:", recall)
print("\nClassification Report:\n")
print(classification_report(y_test, preds))
