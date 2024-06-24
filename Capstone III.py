
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

 
data = pd.read_csv('data.csv')
data_by_artist = pd.read_csv('data_by_artist.csv')
data_by_genres = pd.read_csv('data_by_genres.csv')
data_by_year = pd.read_csv('data_by_year.csv')
data_w_genres = pd.read_csv('data_w_genres.csv')

# Display the first few rows of each dataset
print("data.csv:")
print(data.head())
print("\ndata_by_artist.csv:")
print(data_by_artist.head())
print("\ndata_by_genres.csv:")
print(data_by_genres.head())
print("\ndata_by_year.csv:")
print(data_by_year.head())
print("\ndata_w_genres.csv:")
print(data_w_genres.head())

# Check for missing values and data types
data.info()
data_by_artist.info()
data_by_genres.info()
data_by_year.info()
data_w_genres.info()

# Describe the datasets for statistical overview
data.describe()
data_by_artist.describe()
data_by_genres.describe()
data_by_year.describe()
data_w_genres.describe()

# Drop duplicates if any
data.drop_duplicates(inplace=True)
data_by_artist.drop_duplicates(inplace=True)
data_by_genres.drop_duplicates(inplace=True)
data_by_year.drop_duplicates(inplace=True)
data_w_genres.drop_duplicates(inplace=True)

# Handle missing values by filling or dropping them
data.fillna(data.mean(), inplace=True)
data_by_artist.fillna(data_by_artist.mean(), inplace=True)
data_by_genres.fillna(data_by_genres.mean(), inplace=True)
data_by_year.fillna(data_by_year.mean(), inplace=True)
data_w_genres.fillna(data_w_genres.mean(), inplace=True)
"""
# Select relevant features for modeling
selected_features = [
    'valence', 'year', 'acousticness', 'danceability', 'duration_ms', 
    'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo'
]

data = data[selected_features]
data_by_artist = data_by_artist[selected_features]
data_by_genres = data_by_genres[selected_features]
data_by_year = data_by_year[selected_features]
data_w_genres = data_w_genres[selected_features]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Normalize and standardize each dataset
data_scaled = scaler.fit_transform(data)
data_by_artist_scaled = scaler.fit_transform(data_by_artist)
data_by_genres_scaled = scaler.fit_transform(data_by_genres)
data_by_year_scaled = scaler.fit_transform(data_by_year)
data_w_genres_scaled = scaler.fit_transform(data_w_genres)

# Convert back to DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=selected_features)
data_by_artist_scaled = pd.DataFrame(data_by_artist_scaled, columns=selected_features)
data_by_genres_scaled = pd.DataFrame(data_by_genres_scaled, columns=selected_features)
data_by_year_scaled = pd.DataFrame(data_by_year_scaled, columns=selected_features)
data_w_genres_scaled = pd.DataFrame(data_w_genres_scaled, columns=selected_features)

from sklearn.model_selection import train_test_split

# Split data into training and validation sets
train_data, val_data = train_test_split(data_scaled, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
X_train = train_data.drop('valence', axis=1)
y_train = train_data['valence']
model.fit(X_train, y_train)

X_val = val_data.drop('valence', axis=1)
y_val = val_data['valence']
y_pred = model.predict(X_val)

# Evaluate model performance
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')

 """