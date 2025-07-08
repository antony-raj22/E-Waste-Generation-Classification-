# e_waste_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Sample data (You can replace this with a real dataset)
data = {
    'DeviceType': ['Phone', 'TV', 'Laptop', 'Fridge', 'Printer', 'Phone', 'AC', 'Monitor'],
    'MaterialType': ['Plastic', 'Metal', 'Metal', 'Plastic', 'Mixed', 'Metal', 'Mixed', 'Glass'],
    'Source': ['Household', 'Household', 'Office', 'Household', 'Office', 'Household', 'Industrial', 'Office'],
    'Class': ['Small', 'Large', 'Small', 'Large', 'Medium', 'Small', 'Large', 'Medium']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encoding categorical variables
df_encoded = pd.get_dummies(df[['DeviceType', 'MaterialType', 'Source']])
X = df_encoded
y = df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
