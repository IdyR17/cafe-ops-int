# The model trainer
# after preparing the data, this script will train the ML model
# To measure effectiveness, we will split the data into training set (past data) and test set ("future" data)

from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib

from src.data import prep_data, save_processed_data
from src.model import (
    split_data,
    separate_features_and_target,
    encode_cat_features,
    train_random_forest,
    test_model,
    create_hourly_baseline,
    create_product_baseline,
    run_hourly_growth_experiment,
    run_hourly_trend_experiment,
    run_recent_product_sales_experiment,
)

##### PREPARE DATA #####

# Prepare the data for ML training
hourly_data, product_daily_data = prep_data()

# Save copies of the processed datasets
save_processed_data(hourly_data, product_daily_data)

# Split the data into training and testing subsets
cutoff_date = pd.Timestamp("2023-06-01")

hourly_train, hourly_test = split_data(
    hourly_data,
    cutoff_date,
)

product_daily_train, product_daily_test = split_data(
    product_daily_data,
    cutoff_date,
)


##################################################
##### HOURLY TRANSACTION MODEL #####
##################################################


##### STEP 1: FEATURE SELECTION #####

### Feature Selection ###
# Separate features and target for hourly data
hourly_feature_columns = [
    "store_location",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

hourly_target_column = "transaction_count"

hourly_train_features, hourly_train_target = separate_features_and_target(
    hourly_train,
    hourly_feature_columns,
    hourly_target_column,
)

hourly_test_features, hourly_test_target = separate_features_and_target(
    hourly_test,
    hourly_feature_columns,
    hourly_target_column,
)


##### STEP 2: ENCODE FEATURES #####

# Convert store location into numeric columns for the ML model
hourly_train_features, hourly_test_features = encode_cat_features(
    hourly_train_features,
    hourly_test_features,
    ["store_location"],
)


##### STEP 3: TRAIN MODEL #####

### MODEL TRAINING AND EVALUATION ###
# Train the hourly transaction model
hourly_model = train_random_forest(
    hourly_train_features,
    hourly_train_target,
)


##### STEP 4: MAE EVALUATION #####

####MAE Evaluation####
# Evaluate the hourly transaction model MAE using the June test data
hourly_predictions, hourly_mae = test_model(
    hourly_model,
    hourly_test_features,
    hourly_test_target,
)


##### STEP 5: BASELINE COMPARISON #####

# Create and evaluate the historical-average baseline
hourly_baseline = create_hourly_baseline(
    hourly_train,
    hourly_test,
)

baseline_mae = mean_absolute_error(
    hourly_baseline["transaction_count"],
    hourly_baseline["baseline_prediction"],
)


##### HOURLY MODEL RESULTS #####

print("\n##### HOURLY MODEL RESULTS #####")
print("Historical average baseline MAE:", round(baseline_mae, 2))
print("Random Forest MAE:", round(hourly_mae, 2))


##################################################
##### PRODUCT DEMAND MODEL #####
##################################################


##### STEP 1: FEATURE SELECTION #####

# Features and target for the daily product demand model
product_feature_columns = [
    "store_location",
    "product_id",
    "day_of_week",
    "month",
    "is_weekend",
]

product_target_column = "units_sold"

# Separate features and target for product training data
product_train_features, product_train_target = separate_features_and_target(
    product_daily_train,
    product_feature_columns,
    product_target_column,
)

# Separate features and target for product testing data
product_test_features, product_test_target = separate_features_and_target(
    product_daily_test,
    product_feature_columns,
    product_target_column,
)


##### STEP 2: ENCODE FEATURES #####

product_train_features, product_test_features = encode_cat_features(
    product_train_features,
    product_test_features,
    [
        "store_location",
        "product_id",
    ],
)


##### STEP 3: TRAIN MODEL #####

# Train the daily product demand model
product_model = train_random_forest(
    product_train_features,
    product_train_target,
)


##### STEP 4: MAE EVALUATION #####

# Evaluate the product demand model using the June test data
product_predictions, product_mae = test_model(
    product_model,
    product_test_features,
    product_test_target,
)


##### STEP 5: BASELINE COMPARISON #####

# Compare product model against a historical-average baseline
product_baseline = create_product_baseline(
    product_daily_train,
    product_daily_test,
)

product_baseline_mae = mean_absolute_error(
    product_baseline["units_sold"],
    product_baseline["baseline_prediction"],
)


##### STEP 6: HIGH-VOLUME PRODUCT TEST #####

# MAE for higher-volume product days
high_volume_mask = product_daily_test["units_sold"] >= 10

high_volume_mae = mean_absolute_error(
    product_daily_test.loc[
        high_volume_mask,
        "units_sold",
    ],
    product_predictions[high_volume_mask],
)


##### PRODUCT MODEL RESULTS #####

print("\n##### PRODUCT MODEL RESULTS #####")
print(
    "Product historical average baseline MAE:",
    round(product_baseline_mae, 2),
)

print(
    "Product Random Forest MAE:",
    round(product_mae, 2),
)

print(
    "Product MAE when 10+ units were sold:",
    round(high_volume_mae, 2),
)

##################################################
##### SAVE TRAINED MODELS #####
##################################################


# Save the hourly model and the columns used to train it
joblib.dump(
    {
        "model": hourly_model,
        "feature_columns": hourly_train_features.columns.tolist(),
    },
    "models/hourly_model.joblib",
    compress=3,
)


# Save the product model and the columns used to train it
joblib.dump(
    {
        "model": product_model,
        "feature_columns": product_train_features.columns.tolist(),
    },
    "models/product_model.joblib",
    compress=3,
)


print("\nTrained models saved successfully.")

##################################################
##### OPTIONAL EXPERIMENTS #####
##################################################


##### HOURLY GROWTH EXPERIMENT #####
# Uncomment to test.
#
# Previous results:
# Random Forest MAE: 11.13
# Growth-adjusted Random Forest MAE: 11.78

# run_hourly_growth_experiment(
#     hourly_train,
#     hourly_predictions,
#     hourly_test_target,
# )


##### TREND FEATURE EXPERIMENT #####
# This experiment tests if previous-month average transaction volume
# and previous-month growth improve the Random Forest.
# The trend features produced a slightly higher MAE, so they are not used
# in the final hourly model.
#
# Uncomment to test.
#
# Previous results:
# Original features MAE using March-May training: 11.11
# Trend features MAE using March-May training: 11.36

# run_hourly_trend_experiment(
#     hourly_data,
#     cutoff_date,
# )


##### RECENT PRODUCT SALES EXPERIMENT #####
# Uncomment to test.
#
# Previous results:
# Original product features MAE: 3.17
# Product features + previous 7-day average MAE: 3.13
# Original product model MAE on 10+ unit days: 4.71
# Recent-sales model MAE on 10+ unit days: 4.69

# run_recent_product_sales_experiment(
#     product_daily_data,
#     cutoff_date,
#     high_volume_mae,
# )
