import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# load and split the data
df = pd.read_csv("ASL_numbers_dataset.csv")
X = df.drop("Label", axis=1)
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = KNeighborsClassifier(n_neighbors=70)
model.fit(X_train, y_train)

# to save the model
joblib.dump(model, "gesture_model.pkl")

## model evaluation

y_pred = model.predict(X_test)

print (accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Gesture")
plt.ylabel("Actual Gesture")
plt.show()