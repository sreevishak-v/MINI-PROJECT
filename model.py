import streamlit as st
import joblib

# Load the pre-trained model and vectorizer
model, vectorizer = joblib.load("rf_model_count_vectorizer.pkl")

# Define sentiment labels
sentiment_labels = {'0': 'NOT DEPRESSION', '1': 'MODERATE DEPRESSION', '2': 'SEVERE DEPRESSION'}

# Title of the app
st.title('DEPRESSION CLASSIFICATION')

# Text area for user input
user_input = st.text_area("Enter your text here")

# Predict button
if st.button("Predict"):
    if user_input:
        transformed_input = vectorizer.transform([user_input])
        predicted_sentiment = model.predict(transformed_input)[0]
        predicted_sentiment_label = sentiment_labels[str(predicted_sentiment)]
        st.info(f"Predicted sentiment: {predicted_sentiment_label}")
    else:
        st.warning("Please enter text to classify.")
