import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv("cmake-build-debug/rmse_data.csv", header=None, names=["time", "rmse"])

# Plot the data
plt.plot(data["time"], data["rmse"])
plt.xlabel("Time (ms)")
plt.ylabel("RMSE")
plt.title("RMSE over Time")
plt.grid(True)
plt.show()
