import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os

# Getting rid of those annoying warnings that clutter the output
warnings.filterwarnings('ignore')

# I like this style for plots, looks cleaner
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Let's get started!")

# Try to load the dataset, but don't crash if I forgot to put the file there
try:
    df = pd.read_csv('kc_House_Data.csv')
    print("Got the data. We are good to go.")
except FileNotFoundError:
    print("Uh oh, can't find 'kc_House_Data.csv'. Double check the folder?")
    exit()

# Just want to get a feel for what I'm working with here
print(f"\nOkay, we have {df.shape[0]:,} houses and {df.shape[1]} columns of info.")

print("\nHere's a quick look at the first few rows:")
print(df.head())

print("\nChecking the data types to see if anything looks weird:")
print(df.info())

print("\nLet's look at the basic stats (min, max, averages):")
print(df.describe())

# Always good to check for holes in the data before we get too deep
print("\nScanning for missing values...")
missing_data = df.isnull().sum()
if missing_data.any():
    print("Found some empty spots:")
    for col, count in missing_data[missing_data > 0].items():
        print(f"   {col}: missing {count} times")
else:
    print("Awesome, the dataset is complete. No missing values.")

# Checking if we have the same house listed twice
duplicates = df.duplicated().sum()
print(f"Found {duplicates} duplicate entries.")

print("\nLet's visualize the prices to see the distribution...")
plt.figure(figsize=(15, 5))

# Plotting the raw prices
plt.subplot(1, 2, 1)
plt.hist(df['price'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.title('Price Distribution (Skewed?)')
plt.xlabel('Price ($)')
plt.ylabel('Count')

# Plotting log prices usually helps normalize things
plt.subplot(1, 2, 2)
plt.hist(np.log1p(df['price']), bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
plt.title('Log-Transformed Prices (Much better)')
plt.xlabel('Log(Price)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

print(f"Quick Price Check:")
print(f"Lowest: ${df['price'].min():,}")
print(f"Highest: ${df['price'].max():,}")
print(f"Average: ${df['price'].mean():,.2f}")

# I want to know which features actually matter for the price
print("\nFiguring out what drives the price up or down...")
numeric_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_columns].corr()
price_correlations = correlation_matrix['price'].sort_values(ascending=False)

plt.figure(figsize=(16, 6))

# The big heatmap
plt.subplot(1, 2, 1)
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')

# The top 10 most important features
plt.subplot(1, 2, 2)
top_correlations = price_correlations[1:11]
sns.barplot(x=top_correlations.values, y=top_correlations.index, palette='viridis')
plt.title('Top 10 Price Drivers')
plt.xlabel('Correlation Strength')

plt.tight_layout()
plt.show()

print("\nHere are the winners for price correlation:")
for feature, corr in top_correlations.items():
    print(f"   {feature}: {corr:.3f}")

# Let's drill down into the most interesting features
print("\nVisualizing the relationships for key features...")
key_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade']

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for i, feature in enumerate(key_features):
    # If it's a number with lots of values, scatter plot it. Otherwise, boxplot.
    if df[feature].dtype in ['int64', 'float64'] and len(df[feature].unique()) > 10:
        axes[i].scatter(df[feature], df['price'], alpha=0.5, color='blue')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Price')
        axes[i].set_title(f'{feature} vs Price')
        axes[i].grid(True, alpha=0.3)
    else:
        df.boxplot(column='price', by=feature, ax=axes[i])
        axes[i].set_title(f'Price by {feature}')

plt.tight_layout()
plt.show()

# Time to clean up. I'll make a copy so I don't mess up the original data.
print("\nPreprocessing the data...")
df_clean = df.copy()

# Simple strategy: fill text gaps with the mode, number gaps with median
if missing_data.any():
    print("Patching up those missing values...")
    for column in missing_data[missing_data > 0].index:
        if df_clean[column].dtype == 'object':
            df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
        else:
            df_clean[column].fillna(df_clean[column].median(), inplace=True)

# Feature Engineering: Let's make some useful combos
print("Cooking up some new features...")
df_clean['house_age'] = 2024 - df_clean['yr_built']  # How old is the place?
df_clean['is_renovated'] = (df_clean['yr_renovated'] > 0).astype(int) # Has it been fixed up?
df_clean['price_per_sqft'] = df_clean['price'] / df_clean['sqft_living'] # Value metric
df_clean['total_rooms'] = df_clean['bedrooms'] + df_clean['bathrooms']
df_clean['living_lot_ratio'] = df_clean['sqft_living'] / df_clean['sqft_lot']

print("Added: house_age, is_renovated, price_per_sqft, total_rooms, living_lot_ratio")

# Picking the columns I actually want to train on
print("\nSelecting columns for the model...")
selected_features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    'sqft_basement', 'yr_built', 'zipcode', 'lat', 'long',
    'sqft_living15', 'sqft_lot15', 'house_age', 'is_renovated',
    'total_rooms', 'living_lot_ratio'
]

X = df_clean[selected_features]
y = df_clean['price']

# Zipcode is categorical, so we need to one-hot encode it
X_encoded = pd.get_dummies(X, columns=['zipcode'], drop_first=True)

# Splitting 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"Split complete. Training on {X_train.shape[0]:,} houses, testing on {X_test.shape[0]:,}.")

# Scaling helps linear models behave better
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# I'm going to try a bunch of models and see which one sticks
print("\nStarting the model battle...")
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Running {name}...")

    # Linear models need scaled data, trees usually don't care
    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # Using cross-validation to be sure it's not a fluke
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    # Calculating the error metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_Mean': cv_scores.mean(),
        'Predictions': y_pred,
        'Model': model
    }

    print(f"   -> Accuracy (R2): {r2:.3f}")
    print(f"   -> Avg Error: ${rmse:,.0f}")

# Let's put the scores in a table so we can compare them easily
print("\n--- Final Scoreboard ---")
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[model]['RMSE'] for model in results],
    'MAE': [results[model]['MAE'] for model in results],
    'R2_Score': [results[model]['R2'] for model in results],
    'CV_R2_Score': [results[model]['CV_Mean'] for model in results]
}).sort_values('R2_Score', ascending=False)

print("And the winner is:")
print(performance_df.round(3))

# Visualizing the model comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].barh(performance_df['Model'], performance_df['R2_Score'], color='lightblue')
axes[0, 0].set_title('Accuracy (R2 Score)')

axes[0, 1].barh(performance_df['Model'], performance_df['RMSE'], color='lightcoral')
axes[0, 1].set_title('Average Error (RMSE)')

axes[1, 0].barh(performance_df['Model'], performance_df['MAE'], color='lightgreen')
axes[1, 0].set_title('Mean Absolute Error')

axes[1, 1].barh(performance_df['Model'], performance_df['CV_R2_Score'], color='gold')
axes[1, 1].set_title('Cross Validation Score (Reliability)')

plt.tight_layout()
plt.show()

# Grabbing the best model for deeper analysis
best_model_name = performance_df.iloc[0]['Model']
best_predictions = results[best_model_name]['Predictions']

print(f"\nWe're going with {best_model_name}. It had {performance_df.iloc[0]['R2_Score']:.1%} accuracy.")

print("\nLet's see where the model made mistakes...")
residuals = y_test - best_predictions

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Residual plot - looking for randomness
axes[0, 0].scatter(best_predictions, residuals, alpha=0.5, color='blue')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_title('Residuals vs Predicted')
axes[0, 0].set_ylabel('Error')

# Histogram of errors
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('Error Distribution')

# Q-Q plot to check normality
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

# Actual vs Predicted
axes[1, 1].scatter(y_test, best_predictions, alpha=0.5, color='purple')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red', lw=2)
axes[1, 1].set_title('Actual vs Predicted Prices')

plt.tight_layout()
plt.show()

# Quick sanity check on percentage errors
percentage_errors = np.abs((y_test - best_predictions) / y_test) * 100

print("\nHow close were we typically?")
print(f"   Within 5%: {(percentage_errors <= 5).mean() * 100:.1f}% of the time")
print(f"   Within 10%: {(percentage_errors <= 10).mean() * 100:.1f}% of the time")
print(f"   Within 20%: {(percentage_errors <= 20).mean() * 100:.1f}% of the time")

# Just for fun, let's map the expensive areas
print("\nMapping out the prices...")
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
scatter = plt.scatter(df_clean['long'], df_clean['lat'],
                      c=df_clean['price'], cmap='viridis',
                      alpha=0.6, s=10)
plt.colorbar(scatter, label='Price ($)')
plt.title('Price Heatmap (Location)')

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(df_clean['long'], df_clean['lat'],
                       c=df_clean['price_per_sqft'], cmap='plasma',
                       alpha=0.6, s=10)
plt.colorbar(scatter2, label='Price/SqFt ($)')
plt.title('Value Map (Price per SqFt)')

plt.tight_layout()
plt.show()

print("\n--- Wrap up ---")
print(f"Best Model: {best_model_name}")
print("Done! The model is ready to predict.")