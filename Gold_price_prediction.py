import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Input
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.tensorflow

#  mlflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:15000")
mlflow.set_experiment("Gold_Price_Prediction_ANn")

df = pd.read_csv("gold_price.csv")

print("\n--- CSV CHECK ---")
print("Columns:", df.columns.tolist())

print("\n24K gold price summary:")
print(df["24K Gold (₹/g)"].describe())

print("\nLatest 7 prices:")
print(df[["Date", "24K Gold (₹/g)"]].tail(7))



print("First 5 Rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nDataset size:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

df["Date"] =pd.to_datetime(df["Date"],format="%d-%b-%y")


df= df.sort_values("Date").reset_index(drop=True)

print("\nDAte range:")
print("First Date:",df["Date"].min())
print("Last Date:",df["Date"].max())


print("\nDuplicate dates:",df["Date"].duplicated().sum())


print("\nFirst 5 rows after date conversion:")
print(df.head())




#  Graph:

plt.figure(figsize=(12,6))

plt.plot(
    df["Date"],
    df["24K Gold (₹/g)"],
    label="24K Gold Price"
)

plt.title("24K Gold Price in India Over Time")
plt.xlabel("Date")
plt.ylabel("Price (₹ per gram)")
plt.legend()
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# input -ann model


gold_prices = df["24K Gold (₹/g)"].values

window_size = 7

x =[]
y =[]

for i in range(len(gold_prices)-window_size):
    x.append(gold_prices[i:i + window_size])
    y.append(gold_prices[i+ window_size])





# list into np array
x = np.array(x)
y = np.array(y)


print("\nInput data shape:",x.shape)
print("Target data shape:",y.shape)


print("\n First input(7days):")
print(x[0])

print("\nFirst target(8th day):")
print(y[0])


# train-test-spilt


split_index = int(len(x) *0.8)

x_train = x[:split_index]
x_test = x[split_index:]

y_train = y[:split_index]
y_test  = y[split_index:]


print("Training X:",x_train.shape)
print("Testing  X:",x_test.shape)
print("Training Y:",y_train.shape)
print("Testing  Y:",y_test.shape)

# scaling

x_scaler = MinMaxScaler()

x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

print("Scaled Training X:",x_train_scaled.shape)
print("Scaled Testing X:",x_test_scaled.shape)


y_scaler = MinMaxScaler()

y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1,1))

print("Scaled Training Y:",y_train_scaled.shape)
print("Scaled Testing  Y:",y_test_scaled.shape)


#  Ann model


model = Sequential([
    Input(shape=(7,)),
    Dense(64,activation="relu"),
    Dense(32,activation="relu"),
    Dense(1)
])
model.summary()


# Optimtizer:


model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

# ---------------- START MLFLOW RUN ----------------

with mlflow.start_run(run_name="ANN_Gold_Price_Model") as run:

    # 1. Log model parameters
    mlflow.log_params({
        "model_type": "ANN",
        "window_size": window_size,
        "hidden_layer_1": 64,
        "hidden_layer_2": 32,
        "optimizer": "adam",
        "loss_function": "mean_squared_error",
        "epochs": 100,
        "batch_size": 16,
        "validation_split": 0.2
    })

    # 2. Train the ANN model
    history = model.fit(
        x_train_scaled,
        y_train_scaled,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        shuffle=False
    )

    # 3. Log training and validation metrics
    mlflow.log_metrics({
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_mae": float(
            history.history["mean_absolute_error"][-1]
        ),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_mae": float(
            history.history["val_mean_absolute_error"][-1]
        )
    })

    # 4. Evaluate the model
    test_loss, test_mae = model.evaluate(
        x_test_scaled,
        y_test_scaled,
        verbose=0
    )

    print("Test loss:", test_loss)
    print("Test MAE (scaled):", test_mae)

    # 5. Predict test prices
    y_pred_scaled = model.predict(
        x_test_scaled, verbose=0
    )

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_actual = y_scaler.inverse_transform(y_test_scaled)

    mae_rupees = mean_absolute_error(y_actual, y_pred)

    print("Test MAE in ₹/gram:", mae_rupees)

    # 6. Log test metrics
    mlflow.log_metrics({
        "test_loss_scaled": float(test_loss),
        "test_mae_scaled": float(test_mae),
        "test_mae_rupees_per_gram": float(mae_rupees)
    })

    # 7. Display actual vs predicted prices
    for actual, predicted in zip(y_actual[:10], y_pred[:10]):
        print(
            f"Actual: ₹{actual[0]:.2f} | "
            f"Predicted: ₹{predicted[0]:.2f}"
        )

    # 8. Predict the next gold price
    last_window = gold_prices[-window_size:].reshape(1, -1)

    last_window_scaled = x_scaler.transform(last_window)

    next_price_scaled = model.predict(
        last_window_scaled, verbose=0
    )

    next_price = y_scaler.inverse_transform(
        next_price_scaled
    )[0][0]

    print(f"Predicted next gold price: ₹{next_price:.2f}/gram")

    # 9. Save model and scalers for Flask
    model.save("gold_price_ann.keras")

    joblib.dump(x_scaler, "x_scaler.pkl")
    joblib.dump(y_scaler, "y_scaler.pkl")

    # 10. Log saved files as MLflow artifacts
    mlflow.log_artifact("gold_price_ann.keras")
    mlflow.log_artifact("x_scaler.pkl")
    mlflow.log_artifact("y_scaler.pkl")

    # 11. Print the run ID while the run is active
    print("MLflow Run ID:", run.info.run_id)
    print("MLflow tracking completed.")


