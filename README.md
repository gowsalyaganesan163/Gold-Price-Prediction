# Gold Price Prediction using Artificial Neural Network (ANN)

## Project Overview

This project predicts the next day's 24K gold price (₹ per gram) using historical gold price data and an Artificial Neural Network (ANN).

The model uses the previous **7 days of gold prices** to predict the next price. The project also includes a Flask web application that displays recent gold prices and the predicted next price.

## Features

- Historical gold price data analysis
- Data preprocessing and date sorting
- Exploratory data analysis and price visualization
- 7-day sliding window for prediction
- Artificial Neural Network (ANN) model
- Data normalization using MinMaxScaler
- Model evaluation using Mean Absolute Error (MAE)
- MLflow experiment tracking
- Flask web interface for displaying predictions

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Flask
- MLflow
- Joblib

## Model Architecture

The ANN model consists of:

- Input layer: 7 input values (previous 7 days' gold prices)
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 1 neuron for the predicted gold price

**Model configuration:**

| Parameter | Value |
|---|---|
| Model | Artificial Neural Network (ANN) |
| Input window | 7 days |
| Optimizer | Adam |
| Loss function | Mean Squared Error (MSE) |
| Epochs | 100 |
| Batch size | 16 |
| Validation split | 20% |

## Project Structure

```text
Gold-Price-Prediction/
│
├── app.py
├── Gold_price_prediction.py
├── gold_price.csv
├── gold_price_ann.keras
├── x_scaler.pkl
├── y_scaler.pkl
│
├── templates/
│   └── index.html
│
├── .gitignore
├── README.md
└── requirements.txt
```

## How It Works

1. Load historical gold price data from the CSV file.
2. Convert dates into datetime format and sort the data chronologically.
3. Create training samples using a 7-day sliding window.
4. Split the dataset into training and testing sets.
5. Scale input and target data using MinMaxScaler.
6. Train the ANN model.
7. Evaluate predictions using MAE.
8. Save the trained model and scaler files.
9. Use Flask to display the latest gold prices and next-price prediction.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Gold-Price-Prediction.git
cd Gold-Price-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Web Application

Run the Flask application:

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

## Train the Model

To retrain the ANN and generate the model/scaler files:

```bash
python Gold_price_prediction.py
```

The training script also logs parameters, metrics, and artifacts to MLflow.

## Model Evaluation

The model is evaluated using Mean Absolute Error (MAE), including the MAE converted back to Indian Rupees per gram.

## Disclaimer

This project is for educational and research purposes only. Gold price predictions are estimates generated from historical data and should not be considered financial advice.

## Author

**Gowsalya**
