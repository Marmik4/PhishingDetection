import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# File Paths
absolute_path = os.path.dirname(__file__)
relative_path_DataSet = "DataSets/emails.csv"
relative_path_Model = "Models/phishing_model.pkl"
relative_path_Vectorizer = "Models/tfidf_vectorizer.pkl"
full_path_DataSet = os.path.join(absolute_path, relative_path_DataSet)
full_path_Model = os.path.join(absolute_path, relative_path_Model)
full_path_Vectorizer = os.path.join(absolute_path, relative_path_Vectorizer)

# Load the dataset
data = pd.read_csv(full_path_DataSet)

# Check the first few rows of data to understand its structure
print(data.head(20))

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

# Convert email content to numerical form
X = vectorizer.fit_transform(data['text'])
y = data['label']  # Our target variable (1 for phishing, 0 for legitimate)

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate model performance
print(classification_report(y_test, predictions))

# Save the model
joblib.dump(model, full_path_Model)

# Save the vectorizer
joblib.dump(vectorizer, full_path_Vectorizer)

