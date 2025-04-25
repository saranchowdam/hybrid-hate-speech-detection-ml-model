from flask import Flask, request, jsonify
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
import os  # for accessing environment variables

app = Flask(__name__)

# --------------------------
# Custom Hybrid Model Class
# --------------------------
class HybridModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model1, model2, model3):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def fit(self, X, y):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        self.model3.fit(X, y)
        return self

    def predict(self, X):
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        pred3 = self.model3.predict(X)
        final_pred = mode([pred1, pred2, pred3], axis=0)[0].flatten()
        return final_pred

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
