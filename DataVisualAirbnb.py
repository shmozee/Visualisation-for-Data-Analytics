import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

############### DATA PRE-PROCESSING ###############
print("\nDATA PRE-PROCESSING")

df = pd.read_csv("Airbnb_Open_Data.csv")

df['price'] = df['price'].replace('[\$,]', '', regex=True) 
df['price'] = pd.to_numeric(df['price'], errors='coerce')    
df['service fee'] = df['service fee'].replace('[\$,]', '', regex=True) 
df['service fee'] = pd.to_numeric(df['service fee'], errors='coerce')   

# drop columns
df.drop('NAME', axis=1, inplace=True)
df.drop('host name', axis=1, inplace=True)
df.drop('country', axis=1, inplace=True)
df.drop('country code', axis=1, inplace=True)
df.drop('reviews per month', axis=1, inplace=True)
df.drop('last review', axis=1, inplace=True)
df.drop('house_rules', axis=1, inplace=True)
df.drop('license', axis=1, inplace=True)

#  - Remove any duplicate rows
df.duplicated().value_counts() # calculates duplicates
df.drop_duplicates(inplace=True) # remove duplicates

#  - Replace Nan with unconfirmed in host_identity_verified
df['host_identity_verified'] = df['host_identity_verified'].replace(np.nan, 'unconfirmed')
df['host_identity_verified'].unique()

#  - Fill in missing neighbourhood values 

#  - Replace typos and remove na 
df['neighbourhood group'] = df['neighbourhood group'].replace('brookln', 'Brooklyn')
df['neighbourhood group'] = df['neighbourhood group'].replace('manhatan', 'Manhattan')

# Create mapping from neighbourhood to neighbourhood group based on non-null entries
neighbourhood_to_group = df[df['neighbourhood group'].notna()].drop_duplicates(subset=['neighbourhood'])[['neighbourhood', 'neighbourhood group']]
neighbourhood_to_group = dict(zip(neighbourhood_to_group['neighbourhood'], neighbourhood_to_group['neighbourhood group']))

# Fill missing 'neighbourhood group' values using the mapping
df['neighbourhood group'] = df.apply(
    lambda row: neighbourhood_to_group.get(row['neighbourhood'], row['neighbourhood group']),
    axis=1
)

# Fill missing (NaN) 'minimum nights' with 0
df['minimum nights'] = df['minimum nights'].fillna(0)

# Remove rows where 'minimum nights' is negative
df = df[df['minimum nights'] >= 0]

# Calculate total_cost
df['total_cost'] = df['price'] + df['service fee']

# Reorder columns to place 'total_cost' after 'service fee'
cols = df.columns.tolist()
fee_index = cols.index('service fee')
# Move 'total_cost' to right after 'service fee'
cols.insert(fee_index + 1, cols.pop(cols.index('total_cost')))
df = df[cols]


df.dropna(subset=['neighbourhood'], inplace=True)
df.dropna(subset=['instant_bookable'], inplace=True)
df.dropna(subset=['cancellation_policy'], inplace=True)
df.dropna(subset=['Construction year'], inplace=True)
df.dropna(subset=['price'], inplace=True)
df.dropna(subset=['service fee'], inplace=True)
df.dropna(subset=['review rate number'], inplace=True)
df.dropna(subset=['calculated host listings count'], inplace=True)
df.dropna(subset=['lat'], inplace=True)
df.dropna(subset=['long'], inplace=True)
df = df[(df['availability 365'].notna()) & (df['availability 365'] >= 0)]

df.to_csv("Airbnb_Preprocessed.csv", index=False)


############### DATA VISUALISATION ###############
print("\nDATA VISUALISATION")

# 3.1 Number of Reviews vs Frequency 
df['number of reviews'] = pd.to_numeric(df['number of reviews'], errors='coerce')
df.dropna(subset=['number of reviews'], inplace=True)
plt.figure(figsize=(10, 6))
sns.histplot(
df['number of reviews'],
    bins=50,
    kde=True,
    color='mediumseagreen',
    edgecolor='black',
    line_kws={'color': 'black', 'linewidth': 1.5}
)

plt.title('Distribution of Number of Reviews per Listing', fontsize=14, weight='bold')
plt.xlabel('Number of Reviews', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xlim(0, 200)
plt.grid(True, linestyle='--', linewidth=0.6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

df_verified = df[(df['price'] > 0) & (df['service fee'] > 0)].copy()

# 3.2 Room type vs Total Cost 
df_verified['total_cost'] = df_verified['price'] + df_verified['service fee']
df_verified = df_verified.dropna(subset=['total_cost', 'room type'])
plt.figure(figsize=(10, 6))
sns.violinplot(
    data=df_verified,
    x='room type',
    y='total_cost',
    inner='quartile',
    palette='muted',
    linewidth=1.2
)

plt.ylim(bottom=0)
plt.title('Total Cost Distribution by Room Type (Final, Cleaned)', fontsize=14, weight='bold')
plt.xlabel('Room Type', fontsize=12)
plt.ylabel('Total Cost (USD)', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.6)
plt.tight_layout()

# 3.3 Neighbourhood vs Average Total Price 
avg_price_by_neighbourhood = df.groupby('neighbourhood')['total_cost'].mean().sort_values(ascending=False)
avg_price_df = avg_price_by_neighbourhood.reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=avg_price_df, x='neighbourhood', y='total_cost', palette='viridis')
plt.xticks(visible=False)  # rotate for readability
plt.show()

top10_ag_price_neighbourhoods = avg_price_df.head(10) # top 10
bottom10_ag_price_neighbourhoods = avg_price_df.tail(10) # bottom 10

# Add a label for grouping (used in colour palette)
top10_ag_price_neighbourhoods['label'] = '10 Highest Price Neighborhoods'
bottom10_ag_price_neighbourhoods['label'] = '10 Lowest Price Neighborhoods'

combined_df = pd.concat([top10_ag_price_neighbourhoods, bottom10_ag_price_neighbourhoods])
palette = {
    '10 Highest Price Neighborhoods': '#ADD8E6',  # Light blue
    '10 Lowest Price Neighborhoods': '#FFA07A'   # Light red
}
plt.figure(figsize=(12, 6))
sns.barplot(data=combined_df, x='neighbourhood', y='total_cost', hue='label', palette=palette)
plt.xlabel("Neighbourhood")
plt.ylabel("Price")
plt.xticks(rotation=45, ha='right')  # rotate for readability
plt.legend(title='Neighborhood Category')
plt.tight_layout()
plt.show()

#  3.4 Average Total Cost vs Room Type
avg_price_by_room = df.groupby('room type')['total_cost'].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=avg_price_by_room.index, y=avg_price_by_room.values, palette="viridis")

plt.title('Average Total Cost by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Average Total Cost (USD)')
plt.ylim(700)
plt.tight_layout()
plt.show()

# 3.5 Number of Reviews vs Room Type 
sns.violinplot(x='room type', y='number of reviews', data=df)
plt.title('Number of Reviews by Room Type')
plt.show()

# 3.6 Neighbourhood Group vs Frequency 
plt.figure(figsize=(10,5))
sns.countplot(y='neighbourhood group', data=df, order=df['neighbourhood group'].value_counts().index)
plt.title('Number of Listings by Neighbourhood Group')
plt.show()

cross_tab = pd.crosstab(df['neighbourhood group'], df['room type'])
cross_tab.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title('Room Type Distribution Across Neighbourhood Groups')
plt.ylabel('Number of Listings')
plt.show()

# 3.7 Graphical Distribution of Listings
plt.figure(figsize=(10,6))
plt.scatter(df['long'], df['lat'], c=df['price'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Price ($)')
plt.title('Geographical Distribution of Listings')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# 3.8 Price vs Number of Listings 
plt.figure(figsize=(10, 6))
sns.histplot(df['total_cost'], bins=50, kde=True, color='orange', edgecolor='black')
plt.title("Price Distribution of Airbnb Listings")
plt.xlabel('Price ($)')
plt.ylabel('Number of Listings')
plt.ylim(1500)
plt.grid(True)
plt.show()

# 3.9 Proportions of Room Types
df['room type'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(7,7))
plt.title('Proportion of Room Types')
plt.ylabel('')
plt.show()

# 3.10 Correlation Heat Map 
numeric_df = df.select_dtypes(include='number') # Select numeric columns only
corr_matrix = numeric_df.corr() # Compute correlation matrix

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


############### BIG DATA TECHNIQUES ###############
print("\nBIG DATA TECHNIQUES")

# Read data
df = pd.read_csv('Airbnb_Preprocessed.csv')

# Check the first few rows
df.head()

# Get summary statistics
df.describe()

# Check for missing values
df.isnull().sum()

# Define data processing function
def process_function(chunk):
    """
    Process each chunk of data.
    Replace this with your actual processing logic.
    """
    # Example: Calculate mean of numeric columns
    # Filter out non-numeric columns
    numeric_chunk = chunk.select_dtypes(include=[np.number])
    
    # Example simple processing: calculate summary statistics
    result = pd.DataFrame({
        'mean': numeric_chunk.mean(),
        'median': numeric_chunk.median(),
        'std': numeric_chunk.std()
    })
    
    return result

# Path to CSV file
file_path = 'Airbnb_Preprocessed.csv'

# Process in chunks of 100,000 rows
chunks = pd.read_csv(file_path, chunksize=100000)

# Process each chunk individually
results = []
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}...")
    result = process_function(chunk)
    results.append(result)
    
    # Optional: Display progress
    if i < 3:  # Show first few results
        print(result.head())

# Combine results (stack them vertically)
final_result = pd.concat(results, axis=0)

# Aggregate (e.g., calculate the mean of the summary statistics across chunks)
final_result = final_result.groupby(final_result.index).mean()

print("Final aggregated result:")
print(final_result)

############### MACHINE LEARNING MODELS ###############
print("\nMACHINE LEARNING MODELS")

# Load preprocessed data
df = pd.read_csv("Airbnb_Preprocessed.csv")

# Encode categorical columns
label_encoders = {}
categorical_cols = ['neighbourhood group', 'room type']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
features = ['neighbourhood group', 'room type', 'minimum nights', 'calculated host listings count', 'review rate number']
X = df[features]
y = df['total_cost']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'SVR': SVR()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

# Model Performance Visualization
# Model names
models = ['Linear Regression', 'Random Forest', 'KNN', 'SVR']

# Metric values
rmse_scores = [399.01, 400.64, 430.70, 399.05]
mae_scores = [345.60, 341.96, 362.75, 345.66]
r2_scores = [-0.00, -0.01, -0.17, -0.00]

x = np.arange(len(models))  # label locations
width = 0.25  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
rects1 = ax.bar(x - width, rmse_scores, width, label='RMSE')
rects2 = ax.bar(x, mae_scores, width, label='MAE')
rects3 = ax.bar(x + width, r2_scores, width, label='R² Score')

# Labels and titles
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.legend()

# Add value labels on top of bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

plt.tight_layout()
plt.show()