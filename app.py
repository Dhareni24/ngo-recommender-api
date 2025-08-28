import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS to handle cross-origin requests

# Create the Flask application
app = Flask(__name__)
# Enable CORS for the application
CORS(app)

# --------------------
# Load Data & Model
# --------------------
# This section now reads the data from your uploaded CSV file.
try:
    df = pd.read_csv(r"C:\Users\dhare\Downloads\ngos_dummy_large.csv")
    print("Successfully loaded data from ngos_dummy_large.csv")
except FileNotFoundError:
    print("Error: The file 'ngos_dummy_large.csv' was not found.")
    print("Please make sure the CSV file is in the same directory as this script.")
    df = pd.DataFrame() # Create an empty DataFrame to prevent further errors
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    df = pd.DataFrame()

# --------------------
# Machine Learning Logic
# --------------------
def clean_text(text):
    """
    Cleans a string by converting to lowercase and removing special characters.
    Handles non-string inputs gracefully.
    """
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.strip()

# Only run model training if the DataFrame is not empty
if not df.empty:
    df["clean_desc"] = df["description"].apply(clean_text)
    # Safely handle NaN values in the 'sdg' column before splitting
    df["sdg_label"] = df["sdg"].astype(str).str.split(";", n=1).str[0].str.strip()

    # ML classifier to predict SDG
    # The pipeline will clean the text and then apply TF-IDF and Naive Bayes
    pipeline = Pipeline([
        ("clean", FunctionTransformer(lambda x: [clean_text(str(t)) for t in x])),
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("nb", MultinomialNB())
    ])
    pipeline.fit(df["description"], df["sdg_label"])

    # Recommender
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    ngo_matrix = vectorizer.fit_transform(df["clean_desc"])

def predict_sdg(query):
    """
    Predicts the SDG for a given query using the trained ML pipeline.
    """
    if df.empty:
        return "N/A"
    return pipeline.predict([query])[0]

def recommend_ngos(user_query, location=None, top_k=5):
    """
    Recommends NGOs based on a user query and optional location.
    Calculates cosine similarity between the query and NGO descriptions.
    """
    if df.empty:
        return pd.DataFrame()
    query_clean = clean_text(user_query)
    query_vec = vectorizer.transform([query_clean])
    sims = cosine_similarity(query_vec, ngo_matrix).flatten()
    df["score"] = sims
    
    results = df.copy()
    if location:
        # Filter by location, safely handling case and missing values
        results = results[results["location_city"].str.lower().fillna('') == location.lower()]
    
    return results.sort_values("score", ascending=False).head(top_k)

def plot_results_to_base64(results):
    """
    Generates a bar plot of the recommendations and returns it as a base64 string.
    """
    if results.empty:
        return ""
    
    plt.figure(figsize=(10, 6))
    plt.barh(results["name"], results["score"], color="skyblue")
    plt.xlabel("Relevance Score")
    plt.ylabel("NGOs")
    plt.title("Top NGO Recommendations")
    plt.gca().invert_yaxis()
    
    # Save the plot to a temporary in-memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the buffer content to a base64 string
    base64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close() # Close the plot to free memory
    return base64_string

# --------------------
# API Endpoint
# --------------------
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    API endpoint that receives user input and returns NGO recommendations.
    """
    try:
        data = request.get_json()
        user_query = data.get("query", "")
        user_location = data.get("location", None)
        
        if not user_query:
            return jsonify({"error": "Query is required"}), 400

        # Run the core logic
        predicted_sdg = predict_sdg(user_query)
        recommendations = recommend_ngos(user_query, location=user_location, top_k=5)
        plot_base64 = plot_results_to_base64(recommendations)
        
        # Prepare the response data
        response_data = {
            "predicted_sdg": predicted_sdg,
            "recommendations": recommendations.to_dict('records'),
            "plot_image": plot_base64
        }
        return jsonify(response_data), 200

    except Exception as e:
        # Return a 500 status code with an error message if something goes wrong
        return jsonify({"error": str(e)}), 500

# --------------------
# Main entry point for Heroku/Render
# --------------------
if __name__ == '__main__':
    # Use a dynamic port for hosting services like Heroku or Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
