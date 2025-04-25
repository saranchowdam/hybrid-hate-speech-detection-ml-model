from flask import Flask, request, jsonify
import pickle
import os  # for accessing environment variables

app = Flask(__name__)

# Load the model once when the app starts
with open('hybrid_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the request
    data = request.get_json()
    text = data['text']
    
    # Make a prediction
    prediction = model.predict([text])
    
    # Assign human-readable prediction
    if prediction[0] == 0:
        final_prediction = "Hate Speech"
    else:
        final_prediction = "Non-Hate Speech"

    # Return the prediction as a response
    return jsonify({"final_prediction": final_prediction})
