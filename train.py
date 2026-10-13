# The model trainer
# after preparing the data, this script will train the ML model
# To measure effectiveness, we will split the data into training set (past data) and test set ("future" data)

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


# Prepare the data for ML training
hourly_data, product_daily_data = prep_data()

# Save copies of the processed datasets
save_processed_data(hourly_data, product_daily_data)

# Split the data into training and testing subsets
cutoff_date = pd.Timestamp("2023-06-01")
hourly_train, hourly_test = split_data(hourly_data, cutoff_date)
product_daily_train, product_daily_test = split_data(product_daily_data, cutoff_date)
### Feature Engineering ###
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

# Temporary prints to confirm the data split correctly
print("Hourly training rows:", len(hourly_train))
print("Hourly testing rows:", len(hourly_test))
print("Product training rows:", len(product_daily_train))
print("Product testing rows:", len(product_daily_test))
print("\nHourly columns:")
print(hourly_data.columns.tolist())
# prints for the training features and target
print("\nHourly training features:")
print(hourly_train_features.head())

print("\nHourly training target:")
print(hourly_train_target.head())
