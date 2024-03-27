import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

path = 'Hydra-Movie-Scrape.csv'
df = pd.read_csv(path)
df = df[['Director','Year','Rating']]

print("Model 1 Pipeline:")

if sum(df.isna().any()) == 0:
    print("No Null values")
else:
    # drop null vals
    df = df.dropna()
    print("Null values dropped")

print("Dataset shape:", df.shape)

# Define preprocessing for categorical columns
categorical_features = ['Director', 'Year']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

print("Variables encoded.")

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LinearRegression())])

# Assuming that 'Rating' is what we want to predict
X = df[['Director', 'Year']]
y = df['Rating']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train-test split 80-20%; Linear Regression Model")

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
