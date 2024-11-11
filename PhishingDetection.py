import joblib
import os

# File Paths
absolute_path = os.path.dirname(__file__)
relative_path_Model = "Models/phishing_model.pkl"
relative_path_Vectorizer = "Models/tfidf_vectorizer.pkl"
full_path_Model = os.path.join(absolute_path, relative_path_Model)
full_path_Vectorizer = os.path.join(absolute_path, relative_path_Vectorizer)

# Load the saved model and vectorizer
model = joblib.load(full_path_Model)
vectorizer = joblib.load(full_path_Vectorizer)  # Make sure to save and load the vectorizer as well

# New email text for testing
new_email = ["Dear user, please verify your account to avoid suspension."]

# Transform the new email using the loaded vectorizer
new_email_vectorized = vectorizer.transform(new_email)

# Predict if the new email is phishing or legitimate
prediction = model.predict(new_email_vectorized)

# Output the result
print("Phishing" if prediction[0] == 1 else "Legitimate")