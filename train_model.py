import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
diabetes = pd.read_csv('diabetes_prediction_dataset.csv')

# Data preprocessing
diabetes['age'] = diabetes['age'].round().astype(int)
diabetes['bmi'] = diabetes['bmi'].round().astype(int)
diabetes['HbA1c_level'] = diabetes['HbA1c_level'].round().astype(int)

# Drop smoking_history column
diabetes = diabetes.drop(columns=['smoking_history'])

# Encode categorical variables
le = LabelEncoder()
diabetes['gender'] = le.fit_transform(diabetes['gender'])

# Split into features and target
X = diabetes.drop(columns=['diabetes'])
y = diabetes['diabetes']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
rfc.fit(X_train, y_train)

# Evaluate the model
accuracy = rfc.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model
with open('diabetes.pkl', 'wb') as file:
    pickle.dump(rfc, file)

print("Model trained and saved as 'diabetes.pkl'")