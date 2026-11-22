from flask import Flask, render_template
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

import os

print("APP FILE LOCATION:", os.path.abspath(__file__))
print("TEMPLATE FOLDER:", app.template_folder)
print("EXPECTED HTML LOCATION:", os.path.join(app.root_path, "templates", "index.html"))
print("HTML FILE EXISTS:", os.path.exists(
    os.path.join(app.root_path, "templates", "index.html")
))



# Load the trained ANN model
model = load_model("gold_price_ann.keras")

# Load the saved scalers
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")


print("\n--- SAVED SCALER CHECK ---")
print("Input scaler minimum:", x_scaler.data_min_)
print("Input scaler maximum:", x_scaler.data_max_)

print("Target scaler minimum:", y_scaler.data_min_)
print("Target scaler maximum:", y_scaler.data_max_)




@app.route("/")
def home():
#     return "Flask is responding successfully!"

# if __name__ == "__main__":
#     app.run(
#         host="127.0.0.1",
#         port=5001,
#         debug=False,
#         use_reloader=False
#     )

  

    
    df = pd.read_csv("gold_price.csv")

    
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%b-%y"
    )

    
    df = df.sort_values("Date")


    latest_data = df.tail(7)

    
    prices = latest_data["24K Gold (₹/g)"].to_numpy()

    
    
    input_data = prices.reshape(1, 7)

   
    input_scaled = x_scaler.transform(input_data)

   
    prediction_scaled = model.predict(input_scaled, verbose=0)

    
    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )[0][0]

    print("Prediction in rupees:", prediction)
    
    latest_prices = []

    for _, row in latest_data.iterrows():
        latest_prices.append({
            "date": row["Date"].strftime("%d-%b-%Y"),
            "price": row["24K Gold (₹/g)"]
        })

    return render_template(
        "index.html",
        latest_prices=latest_prices,
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)