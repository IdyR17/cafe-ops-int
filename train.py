# The model trainer
# after preparing the data, this script will train the ML model
# To measure effectiveness, we will split the data into training set (past data) and test set ("future" data)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
from src.data import prep_data, save_processed_data


# Split data into training and testing subsets based on date
# by using the transaction_date column and a cutoff date
# For this purpose, we will train on data before June 2023 and test on data using June.
def split_data(data, cutoff_date):
    train_data = data[data["transaction_date"] < cutoff_date].copy()

    test_data = data[data["transaction_date"] >= cutoff_date].copy()

    return train_data, test_data


# Feature engineering. Function to separate features and target from the dataframes.
# For the hourly data, the target is transaction_count, for the product daily data, the target is units_sold,
# The features are most of the rest of the columns.
def separate_features_and_target(data, feature_columns, target_column):
    features = data[
        feature_columns
    ].copy()  # make sure to copy and not just reference the original dataframe
    target = data[target_column].copy()
    return features, target


# Convert text-based columns into numeric columns using yes/no encoding (store_location)
def encode_cat_features(
    train_features,
    test_features,
    cat_columns,
):
    encoded_train_features = pd.get_dummies(
        train_features,
        columns=cat_columns,
        dtype=int,
    )

    encoded_test_features = pd.get_dummies(
        test_features,
        columns=cat_columns,
        dtype=int,
    )

    # Make sure the test data has the exact same columns as the training data
    encoded_test_features = encoded_test_features.reindex(
        columns=encoded_train_features.columns,
        fill_value=0,
    )

    return encoded_train_features, encoded_test_features


##### TRAINING THE MODEL #####
# Train Random Forest regression model
def train_random_forest(train_features, train_target):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    model.fit(train_features, train_target)
    print("\nHourly Random Forest trained successfully.")

    return model


# Tests the trained model and calculate MAE
def test_model(model, test_features, test_target):
    predictions = model.predict(test_features)

    mae = mean_absolute_error(test_target, predictions)

    return predictions, mae


# Prepare the data for ML training
hourly_data, product_daily_data = prep_data()

# Save copies of the processed datasets
save_processed_data(hourly_data, product_daily_data)

# Split the data into training and testing subsets
cutoff_date = pd.Timestamp("2023-06-01")
hourly_train, hourly_test = split_data(hourly_data, cutoff_date)
product_daily_train, product_daily_test = split_data(product_daily_data, cutoff_date)
### Feature Selection ###
# Separate features and target for hourly data
hourly_train_features, hourly_train_target = separate_features_and_target(
    hourly_train,
    [
        "store_location",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ],
    "transaction_count",
)
hourly_test_features, hourly_test_target = separate_features_and_target(
    hourly_test,
    [
        "store_location",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ],
    "transaction_count",
)

# Convert store location into numeric columns for the ML model
hourly_train_features, hourly_test_features = encode_cat_features(
    hourly_train_features,
    hourly_test_features,
    ["store_location"],
)

### MODEL TRAINING AND EVALUATION ###
# Train the hourly transaction model
hourly_model = train_random_forest(
    hourly_train_features,
    hourly_train_target,
)
####MAE Evaluation####
# Evaluate the hourly transaction model MAE using the June test data
hourly_predictions, hourly_mae = test_model(
    hourly_model,
    hourly_test_features,
    hourly_test_target,
)

print("\nHourly model MAE:", round(hourly_mae, 2))

# Compare some actual transaction counts to the model predictions
prediction_comparison = hourly_test[
    ["store_location", "transaction_date", "hour", "transaction_count"]
].copy()

prediction_comparison["predicted_transactions"] = hourly_predictions

print("\nSample hourly predictions:")
print(prediction_comparison.head(10))

# Temporary prints to confirm encoding worked as expected
print("\nEncoded hourly training features:")
print(hourly_train_features.head())

print("\nEncoded hourly feature columns:")
print(hourly_train_features.columns.tolist())
