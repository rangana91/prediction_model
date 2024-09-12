from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the saved model, encoders, and scaler
xgb_model = joblib.load('xgb_model.pkl')
le_gender = joblib.load('le_gender.pkl')
le_item_purchased = joblib.load('le_item_purchased.pkl')
scaler = joblib.load('scaler.pkl')

# Create Flask app
app = Flask(__name__)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict_top_5_items():
    # Get the request data (age and gender) from the JSON body
    data = request.json
    
    # Extract age and gender
    age = data.get('age')
    gender = data.get('gender')
    
    # Ensure age and gender are provided
    if age is None or gender is None:
        return jsonify({'error': 'Age and gender must be provided'}), 400
    
    # Create a new user input with the provided features
    new_user = pd.DataFrame({
        'Age': [age],
        'Gender': le_gender.transform([gender]),  # Encode the gender using its respective encoder
    })
    
    # Scale the new user's input
    new_user_scaled = scaler.transform(new_user)
    
    # Predict probabilities for each item
    item_probabilities = xgb_model.predict_proba(new_user_scaled)
    
    # Get the top 5 items with the highest probabilities
    top_5_indices = np.argsort(item_probabilities[0])[::-1][:5]
    
    # Convert the indices back to the original item names using the label encoder for items
    top_5_items = le_item_purchased.inverse_transform(top_5_indices)
    
    # Return the top 5 items as JSON
    return jsonify({'top_5_items': top_5_items.tolist()})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
