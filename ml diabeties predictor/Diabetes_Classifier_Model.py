import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from joblib import dump
import warnings

warnings.filterwarnings('ignore')
pima_df = pd.read_csv("diabetes.csv")  # Ensure the file is in the same directory or provide the full path.
print("Dataset loaded successfully.")
print("First 5 rows of the dataset:")
print(pima_df.head())
print("\nDataset Summary:")
print(pima_df.describe())
print("\nDataset Information:")
print(pima_df.info())
print("\nNull Values in the Dataset:")
print(pima_df.isnull().sum())
# Check if 'Outcome' is in the dataset and contains valid values
print("\nUnique values in 'Outcome' column:", pima_df['Outcome'].unique())
plt.figure(figsize=(8, 5))
pima_df['Age'].hist(color='skyblue', edgecolor='black')
plt.title("Histogram of Age", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()
# Outcome distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=pima_df, palette='pastel')
plt.title('Distribution of Outcome', fontsize=16)
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Non-Diabetic', 'Diabetic'])
plt.show()
sns.pairplot(pima_df, hue='Outcome', corner=True, palette='Set2')
plt.show()
# Correlation Heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(pima_df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
plt.show()
# Import the necessary class
from sklearn.impute import SimpleImputer

# Imputation: Replace 0 values in certain columns with the column median
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='median')
pima_df[columns_to_impute] = imputer.fit_transform(pima_df[columns_to_impute])
# Verify no zero values remain in these columns
print("\nAfter imputation, minimum values in relevant columns:")
print(pima_df[columns_to_impute].min())
# Partitioning the data set
X = pima_df.drop('Outcome', axis=1)
y = pima_df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"\nTraining data shape: {X_train.shape}, Test data shape: {X_test.shape}")
# Building the Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=1000, random_state=42)
# Training the model
print("Training the Random Forest Classifier...")
rfc.fit(X_train, y_train)
# Predictions
preds_rfc = rfc.predict(X_test)
print(f"First 10 predictions: {preds_rfc[:10]}")  # Check the first 10 predictions
print(f"First 10 true labels: {y_test[:10].values}")  # Check the first 10 true labels
# Model Evaluation
accuracy = accuracy_score(y_test, preds_rfc)
print(f"\nAccuracy of the Random Forest model: {accuracy * 100:.2f}%")

# Detailed Evaluation
print("\nRandom Forest Classifier's Evaluation Report:")
print(classification_report(y_test, preds_rfc))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds_rfc))
# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix

# ... (rest of your code) ...

# ROC Curve and AUC
y_proba = rfc.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)  # Now roc_curve is defined
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc="lower right")
plt.show()
# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score # Import cross_val_score

# ... (rest of your code) ...

# Cross-validation Accuracy
cv_scores = cross_val_score(rfc, X, y, cv=5)
print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix, matthews_corrcoef # Import matthews_corrcoef here


# ... (rest of your code) ...

# Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test, preds_rfc)
print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.2f}")

# Feature Importance
plt.figure(figsize=(10, 6))
importances = rfc.feature_importances_
feature_names = X.columns
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.show()

# Saving the model
MODEL_NAME = "C:/ml diabeties predictor/Diabetes_Model1.pkl"  # Use full path to ensure correct location
print("\nSaving model...")
dump(rfc, MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")
