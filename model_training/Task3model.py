# Task3model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

iris = pd.read_csv(r"E:\CodSoftIntern\datasets\IRIS.csv")
print("Dataset Overview:")
print(iris.head())

X = iris.drop('species', axis=1)
y = iris['species']

species_mapping = {name: idx for idx, name in enumerate(y.unique())}
y_encoded = y.map(species_mapping)

joblib.dump(species_mapping, 'iris_species_mapping.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='linear', random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=[k for k, v in sorted(species_mapping.items(), key=lambda x: x[1])]))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=species_mapping.keys(),
            yticklabels=species_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

joblib.dump(model, 'iris_svc_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')
print("\nModel and scaler saved successfully")