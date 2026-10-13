# The model trainer
# after preparing the data, this script will train the ML model by splitting
# the data into training set (past data) and test set ("future" data)

import pandas as pd
from src.data import prep_data, save_processed_data

# Prepare the data for ML training
hourly_data, product_daily_data = prep_data()

# Save copies of the processed datasets
save_processed_data(hourly_data, product_daily_data)


# Temporary prints to confirm the data loaded correctly
print("Hourly training rows:", len(hourly_data))
print("Product training rows:", len(product_daily_data))
print("\nHourly columns:")
print(hourly_data.columns.tolist())
