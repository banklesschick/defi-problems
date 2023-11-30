import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data from GitHub URL
url = 'https://raw.githubusercontent.com/banklesschick/defi-problems/main/aggregate_steth_bridge_activity.csv'
df = pd.read_csv(url)

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Assuming you have a DataFrame named 'df' with features and labels
# Replace 'features' and 'labels' with your actual column names

# Extract features and labels
features = df[['AMOUNT_USD', 'TOKEN_ADDRESS', 'feature3']]  # Replace with your feature columns
labels = df['DESTINATION_CHAIN']  # Replace with your label column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))

