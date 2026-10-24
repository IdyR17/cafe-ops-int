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
    return model


# Tests the trained model and calculate MAE
def test_model(model, test_features, test_target):
    predictions = model.predict(test_features)

    mae = mean_absolute_error(test_target, predictions)

    return predictions, mae


# Create baseline predictions using historical average transactions
def create_hourly_baseline(train_data, test_data):

    # Average for each store, day of week, and hour
    historical_averages = (
        train_data.groupby(["store_location", "day_of_week", "hour"])[
            "transaction_count"
        ]
        .mean()
        .reset_index(name="baseline_prediction")
    )

    # Match those averages to the test data
    baseline_data = test_data.merge(
        historical_averages,
        on=["store_location", "day_of_week", "hour"],
        how="left",
    )

    # Create a backup average using just store and hour
    store_hour_averages = (
        train_data.groupby(["store_location", "hour"])["transaction_count"]
        .mean()
        .reset_index(name="store_hour_average")
    )

    baseline_data = baseline_data.merge(
        store_hour_averages,
        on=["store_location", "hour"],
        how="left",
    )

    # Use the store/hour average when the detailed average is missing
    baseline_data["baseline_prediction"] = baseline_data["baseline_prediction"].fillna(
        baseline_data["store_hour_average"]
    )

    # Final fallback: overall training average
    baseline_data["baseline_prediction"] = baseline_data["baseline_prediction"].fillna(
        train_data["transaction_count"].mean()
    )

    return baseline_data


# Create baseline predictions using historical average product sales
def create_product_baseline(train_data, test_data):

    # Average units sold for each store, product, and day of week
    historical_averages = (
        train_data.groupby(["store_location", "product_id", "day_of_week"])[
            "units_sold"
        ]
        .mean()
        .reset_index(name="baseline_prediction")
    )

    baseline_data = test_data.merge(
        historical_averages,
        on=["store_location", "product_id", "day_of_week"],
        how="left",
    )

    # Backup average using store and product
    store_product_averages = (
        train_data.groupby(["store_location", "product_id"])["units_sold"]
        .mean()
        .reset_index(name="store_product_average")
    )

    baseline_data = baseline_data.merge(
        store_product_averages,
        on=["store_location", "product_id"],
        how="left",
    )

    # Use store/product average if detailed average is missing
    baseline_data["baseline_prediction"] = baseline_data["baseline_prediction"].fillna(
        baseline_data["store_product_average"]
    )

    # Final fallback
    baseline_data["baseline_prediction"] = baseline_data["baseline_prediction"].fillna(
        train_data["units_sold"].mean()
    )

    return baseline_data


# Calculate the average monthly transaction growth using training data only
def calculate_average_monthly_growth(train_data):
    monthly_averages = (
        train_data.groupby("month")["transaction_count"].mean().sort_index()
    )

    monthly_growth_rates = monthly_averages.pct_change().dropna()

    average_growth_rate = monthly_growth_rates.mean()

    return average_growth_rate


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

# Calculate historical monthly growth using training data only
average_monthly_growth = calculate_average_monthly_growth(hourly_train)

print(
    "\nAverage monthly growth from training data:",
    round(average_monthly_growth * 100, 2),
    "%",
)


####MAE Evaluation####
# Evaluate the hourly transaction model MAE using the June test data


hourly_predictions, hourly_mae = test_model(
    hourly_model,
    hourly_test_features,
    hourly_test_target,
)

# Adjust Random Forest predictions using historical growth
adjusted_hourly_predictions = hourly_predictions * (1 + average_monthly_growth)

print("\nHourly model MAE:", round(hourly_mae, 2))
adjusted_hourly_mae = mean_absolute_error(
    hourly_test_target,
    adjusted_hourly_predictions,
)

print("Growth-adjusted Random Forest MAE:", round(adjusted_hourly_mae, 2))


# Create and evaluate the historical-average baseline
hourly_baseline = create_hourly_baseline(
    hourly_train,
    hourly_test,
)

baseline_mae = mean_absolute_error(
    hourly_baseline["transaction_count"],
    hourly_baseline["baseline_prediction"],
)

print("\nHistorical average baseline MAE:", round(baseline_mae, 2))
print("Random Forest MAE:", round(hourly_mae, 2))

##### TREND FEATURE EXPERIMENT #####
# This experiment tests if previous-month average transaction volume
# and previous-month growth improve the Random Forest.
# The trend features produced a slightly higher MAE, so they are not used
# in the final hourly model.

trend_hourly_data = hourly_data.dropna(
    subset=[
        "previous_month_avg",
        "previous_month_growth",
    ]
).copy()

trend_hourly_train, trend_hourly_test = split_data(
    trend_hourly_data,
    cutoff_date,
)

original_feature_columns = [
    "store_location",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

trend_feature_columns = [
    "store_location",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "previous_month_avg",
    "previous_month_growth",
]

original_train_features, original_train_target = separate_features_and_target(
    trend_hourly_train,
    original_feature_columns,
    "transaction_count",
)

original_test_features, original_test_target = separate_features_and_target(
    trend_hourly_test,
    original_feature_columns,
    "transaction_count",
)

trend_train_features, trend_train_target = separate_features_and_target(
    trend_hourly_train,
    trend_feature_columns,
    "transaction_count",
)

trend_test_features, trend_test_target = separate_features_and_target(
    trend_hourly_test,
    trend_feature_columns,
    "transaction_count",
)

original_train_features, original_test_features = encode_cat_features(
    original_train_features,
    original_test_features,
    ["store_location"],
)

trend_train_features, trend_test_features = encode_cat_features(
    trend_train_features,
    trend_test_features,
    ["store_location"],
)

original_comparison_model = train_random_forest(
    original_train_features,
    original_train_target,
)

trend_model = train_random_forest(
    trend_train_features,
    trend_train_target,
)

original_comparison_predictions, original_comparison_mae = test_model(
    original_comparison_model,
    original_test_features,
    original_test_target,
)

trend_predictions, trend_mae = test_model(
    trend_model,
    trend_test_features,
    trend_test_target,
)

print(
    "\nOriginal features MAE using March-May training:",
    round(original_comparison_mae, 2),
)

print(
    "Trend features MAE using March-May training:",
    round(trend_mae, 2),
)

##### PRODUCT DEMAND MODEL #####

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


print("\nProduct training features:")
print(product_train_features.head())

print("\nProduct training target:")
print(product_train_target.head())

product_train_features, product_test_features = encode_cat_features(
    product_train_features,
    product_test_features,
    [
        "store_location",
        "product_id",
    ],
)

print("\nEncoded product training features:")
print(product_train_features.head())

print("\nNumber of product model features:")
print(len(product_train_features.columns))

# Train the daily product demand model
product_model = train_random_forest(
    product_train_features,
    product_train_target,
)

print("\nProduct Random Forest trained successfully.")

# Evaluate the product demand model using the June test data
product_predictions, product_mae = test_model(
    product_model,
    product_test_features,
    product_test_target,
)

print("\nProduct model MAE:", round(product_mae, 2))


# Compare actual product sales to model predictions
product_prediction_comparison = product_daily_test[
    [
        "store_location",
        "transaction_date",
        "product_id",
        "product_detail",
        "units_sold",
    ]
].copy()

product_prediction_comparison["predicted_units"] = product_predictions

print("\nSample product predictions:")
print(product_prediction_comparison.head(15))

# Calculate the error for each product prediction
product_prediction_comparison["absolute_error"] = (
    product_prediction_comparison["units_sold"]
    - product_prediction_comparison["predicted_units"]
).abs()

# Show some of the highest-volume product days
high_volume_predictions = product_prediction_comparison.sort_values(
    "units_sold",
    ascending=False,
)

print("\nSample high-volume product predictions:")
print(high_volume_predictions.head(20))

# Compare product model against a historical-average baseline
product_baseline = create_product_baseline(
    product_daily_train,
    product_daily_test,
)

product_baseline_mae = mean_absolute_error(
    product_baseline["units_sold"],
    product_baseline["baseline_prediction"],
)

print(
    "\nProduct historical average baseline MAE:",
    round(product_baseline_mae, 2),
)

print(
    "Product Random Forest MAE:",
    round(product_mae, 2),
)
