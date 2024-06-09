from flask import Flask, request, jsonify
from flask.templating import render_template
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

# Define NeuralNetwork3 class
class NeuralNetwork3(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32):
        super(NeuralNetwork3, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.output = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)  # No sigmoid here because BCEWithLogitsLoss includes it
        return x

app = Flask(__name__)

# Load the model
model_save_path = 'models/best_neural_network_model.pth'
model = torch.load(model_save_path)
model.eval()  # Set the model to evaluation mode

# Load the scaler
scaler = StandardScaler()

# Load the data
data = pd.read_csv('fraud_oracle.csv')

# One hot encode categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Define target and feature variables
y = data['FraudFound_P'].values
X = data.drop('FraudFound_P', axis=1).values

# Normalize inputs
X = scaler.fit_transform(X)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Convert to tensors and move to device
device = 'cpu'
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

def preprocess_input(input_string):
    input_values = input_string.split()
    float_values = [float(value) for value in input_values]
    return np.array(float_values).reshape(1, -1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form  # Access form data
    input_data = preprocess_input(data['input'])  # Preprocess input string
    features = scaler.transform(input_data)
    features = torch.tensor(features, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        output = model(features)
        prediction = (output > 0.5).float().item()

    return jsonify({'message': 'The predicted class is:', 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
