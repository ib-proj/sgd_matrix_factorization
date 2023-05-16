import matplotlib.pyplot as plt
import pandas as pd

# Load RMSE values from CSV file
df = pd.read_csv("cmake-build-debug/dsgd_results.csv")

# Extract time and RMSE values
time = df["Time"]
rmse = df["RMSE"]

# Plot RMSE values over time
plt.plot(time, rmse)
plt.xlabel("Time")
plt.ylabel("RMSE")
plt.title("RMSE vs. Time DSGD")
plt.show()
