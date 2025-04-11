# -*- coding: utf-8 -*-
"""Diabetes Prediction (Balanced)"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load dataset
diabetes = pd.read_csv('diabetes_prediction_dataset.csv')

# Round off and convert types
diabetes['age'] = diabetes['age'].round().astype(int)
diabetes['bmi'] = diabetes['bmi'].round().astype(int)
diabetes['HbA1c_level'] = diabetes['HbA1c_level'].round().astype(int)

# Drop categorical column with too many categories
diabetes = diabetes.drop(columns=['smoking_history'])

# Encode gender
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
diabetes['gender'] = le.fit_transform(diabetes['gender'])

# Check class balance
print("Class distribution:")
print(diabetes['diabetes'].value_counts())

# Features and target
X = diabetes.drop(columns=['diabetes'])
y = diabetes['diabetes']

# Apply SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Random Forest with class_weight
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')
rfc.fit(X_train, y_train)

# Prediction and evaluation
from sklearn.metrics import confusion_matrix, classification_report

y_pred = rfc.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
import pickle
filename = 'diabetes.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rfc, file)

print(f'\nModel saved to {filename}')
