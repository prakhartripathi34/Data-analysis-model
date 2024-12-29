import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the CSV file
file_path = 'refrigeration_service_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Extract the month (as Year-Month for grouping)
data['Month'] = data['Date'].dt.to_period('M')

# Group by 'Month' and 'Failure_Type', summing the Service_Cost
monthly_costs = data.groupby(['Month', 'Failure_Type'])['Service_Cost'].sum().reset_index()

# Pivot the data to have 'Failure_Type' as columns (for multi-line plotting)
pivoted_data = monthly_costs.pivot(index='Month', columns='Failure_Type', values='Service_Cost').fillna(0)

# Convert 'Month' back to timestamp for plotting
pivoted_data.index = pivoted_data.index.to_timestamp()

# Plotting Monthly Service Cost Trends by Failure Type
plt.figure(figsize=(14, 8))
for failure_type in pivoted_data.columns:
    plt.plot(pivoted_data.index, pivoted_data[failure_type], label=failure_type, marker='o', linestyle='-', linewidth=2)

plt.title('Monthly Service Cost Trends by Failure Type', fontsize=20, fontweight='bold')
plt.xlabel('Month', fontsize=16)
plt.ylabel('Service Cost', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Group by 'Month' and 'Failure_Type', counting the frequency of each failure type
monthly_frequency = data.groupby(['Month', 'Failure_Type']).size().reset_index(name='Frequency')

# Pivot the data to have 'Failure_Type' as columns (for multi-line plotting)
pivoted_frequency = monthly_frequency.pivot(index='Month', columns='Failure_Type', values='Frequency').fillna(0)

# Convert 'Month' back to timestamp for plotting
pivoted_frequency.index = pivoted_frequency.index.to_timestamp()

# Plotting Monthly Frequency of Failures by Type
plt.figure(figsize=(14, 8))
for failure_type in pivoted_frequency.columns:
    plt.plot(pivoted_frequency.index, pivoted_frequency[failure_type], label=failure_type, marker='o', linestyle='-', linewidth=2)

plt.title('Monthly Frequency of Failures by Type', fontsize=20, fontweight='bold')
plt.xlabel('Month', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Handle missing values in Service_Cost and Repair_Time
data['Service_Cost'] = data['Service_Cost'].fillna(data['Service_Cost'].mean())
data['Repair_Time'] = data['Repair_Time'].fillna(data['Repair_Time'].mean())

# Ensure correct data types for numerical and categorical columns
data['Service_Cost'] = data['Service_Cost'].astype(float)
data['Repair_Time'] = data['Repair_Time'].astype(float)
data['Failure_Type'] = data['Failure_Type'].astype(str)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
data['Cluster'] = kmeans.fit_predict(data[['Service_Cost', 'Repair_Time']])

# Plotting Scatter Plot with Clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Service_Cost', y='Repair_Time', hue='Cluster', palette='Set2', data=data, s=100, edgecolor='k', alpha=0.7)
plt.title('K-Means Clustering of Service Cost and Repair Time', fontsize=20, fontweight='bold')
plt.xlabel('Service Cost', fontsize=16)
plt.ylabel('Repair Time', fontsize=16)
plt.legend(title='Cluster', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Analyze Cluster Characteristics
cluster_summary = data.groupby('Cluster')[['Service_Cost', 'Repair_Time']].mean()
print("Cluster Characteristics:")
print(cluster_summary)

# Percentage of Failure Types in Each Cluster
cluster_failure_percentages = data.groupby(['Cluster', 'Failure_Type']).size().unstack(fill_value=0)
cluster_failure_percentages = cluster_failure_percentages.div(cluster_failure_percentages.sum(axis=1), axis=0) * 100
print("Percentage of Failure Types in Each Cluster:")
print(cluster_failure_percentages)

# Plotting Heatmap of Failure_Type vs. Aggregated Metrics
pivot_table = data.pivot_table(index='Failure_Type', values=['Service_Cost', 'Repair_Time'], aggfunc='mean')
plt.figure(figsize=(10,10))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, linecolor='black')
plt.title('Heatmap of Average Service Cost and Repair Time by Failure Type', fontsize=20, fontweight='bold')
plt.xlabel('Metric', fontsize=16)
plt.ylabel('Failure Type', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Plotting Pie Chart for Percentage of Failure Reasons
failure_type_counts = data['Failure_Type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(failure_type_counts, labels=failure_type_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'), textprops={'fontsize': 12})
plt.title('Percentage of Failure Reasons', fontsize=20, fontweight='bold')
plt.tight_layout()

# Plotting Pie Chart for Failure Percentage by Device ID
device_failures = data['Device_ID'].value_counts()
total_failures = device_failures.sum()
failure_percentages = (device_failures / total_failures) * 100

plt.figure(figsize=(8, 8))
plt.pie(failure_percentages, labels=failure_percentages.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'], textprops={'fontsize': 12})
plt.title('Failure Percentage by Device ID', fontsize=20, fontweight='bold')
plt.tight_layout()

# Correlation Analysis: Service_Cost vs. Repair_Time by Failure_Type
failure_type_groups = data.groupby('Failure_Type')
correlations_by_failure_type = failure_type_groups[['Service_Cost', 'Repair_Time']].corr().iloc[0::2, -1]

# Display the correlations
print("Correlations between Service_Cost and Repair_Time for each Failure_Type:")
print(correlations_by_failure_type)

# Top 3 most common causes of breakdowns
top_3_failures = data['Failure_Type'].value_counts().head(3)
print("Top 3 Most Common Causes of Breakdowns:")
print(top_3_failures)

# Calculate the average repair time for each failure type
average_repair_time = data.groupby('Failure_Type')['Repair_Time'].mean()

# Calculate the frequency of failures for each failure type
failure_frequency = data['Failure_Type'].value_counts()

# Combine the metrics
combined_metric = failure_frequency * average_repair_time

# Identify the failure types with the highest combined metric
top_problematic_failures = combined_metric.sort_values(ascending=False).head(3)
print("Top Problematic Failure Types (based on frequency and repair time):")
print(top_problematic_failures)

# Plotting the combined metric for all failure types
plt.figure(figsize=(14, 8))
combined_metric.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Combined Metric of Frequency and Repair Time for Failure Types', fontsize=20, fontweight='bold')
plt.xlabel('Failure Type', fontsize=16)
plt.ylabel('Combined Metric (Frequency * Repair Time)', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Plotting Total Service Cost for Each Month
total_monthly_costs = data.groupby('Month')['Service_Cost'].sum()
plt.figure(figsize=(14, 8))
total_monthly_costs.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Total Service Cost for Each Month', fontsize=20, fontweight='bold')
plt.xlabel('Month', fontsize=16)
plt.ylabel('Total Service Cost', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show all plots
plt.show()
