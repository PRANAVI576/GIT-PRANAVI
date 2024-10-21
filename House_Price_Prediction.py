# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Load the dataset
data = pd.read_csv('house_data.csv', sep='\t')  # Adjust 'sep' if needed

# Step 3: Inspect the DataFrame
print("Columns in the dataset:")
print(data.columns)

# Optionally, strip any whitespace from the column names
data.columns = data.columns.str.strip()

# Step 4: Data Preprocessing
# Check if the required columns exist
required_columns = ['size', 'bedrooms', 'age', 'price']
if all(col in data.columns for col in required_columns):
    X = data[['size', 'bedrooms', 'age']]
    y = data['price']
else:
    raise KeyError(f"Missing columns: {set(required_columns) - set(data.columns)}")

# Step 5: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create a Linear Regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 8: Create comparison DataFrame
comparison = pd.DataFrame({'Actual Price': y_test.values, 'Predicted Price': y_pred})
pd.set_option('display.float_format', '{:.2f}'.format)

print("\nHouse Price Predictions vs Actual Prices:")
print(comparison.head())  # Display the first few predictions for comparison



