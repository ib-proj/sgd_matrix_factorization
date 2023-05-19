import matplotlib.pyplot as plt
import pandas as pd

# Load RMSE values from CSV file

df_1 = pd.read_csv("cmake-build-debug/dsgd_results.csv", on_bad_lines='skip')
df_2 = pd.read_csv("cmake-build-debug/dsgd_results_hot.csv", on_bad_lines='skip')
df_3 = pd.read_csv("cmake-build-debug/fsgd_results.csv", on_bad_lines='skip')



result= pd.concat([df_1,df_2,df_3])

result['Time']=pd.to_numeric(result['Time'], errors='coerce').astype('Int64')
result=result[result['Time']<5000]

grouped_df = result.groupby('method')

# Plot the RMSE over time for each method on the same graph
plt.figure(figsize=(10, 6))
for method, group in grouped_df:
    plt.plot(group['Time'], group['RMSE'], label=method)

plt.xlabel('Time(ms)')
plt.ylabel('RMSE')
plt.title('RMSE over Time by Method')
plt.legend()
plt.grid(True)
plt.show()