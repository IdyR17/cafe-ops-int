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


# Prepare the data for ML training
hourly_data, product_daily_data = prep_data()

# Save copies of the processed datasets
save_processed_data(hourly_data, product_daily_data)

# Split the data into training and testing subsets
cutoff_date = pd.Timestamp("2023-06-01")
hourly_train, hourly_test = split_data(hourly_data, cutoff_date)
product_daily_train, product_daily_test = split_data(product_daily_data, cutoff_date)

# Temporary prints to confirm the data split correctly
print("Hourly training rows:", len(hourly_train))
print("Hourly testing rows:", len(hourly_test))
print("Product training rows:", len(product_daily_train))
print("Product testing rows:", len(product_daily_test))
print("\nHourly columns:")
print(hourly_data.columns.tolist())

# Temporary prints to confirm training and testing ranges are accurate

print(
    "\nHourly training dates:",
    hourly_train["transaction_date"].min(),
    "to",
    hourly_train["transaction_date"].max(),
)

print(
    "Hourly testing dates:",
    hourly_test["transaction_date"].min(),
    "to",
    hourly_test["transaction_date"].max(),
)
