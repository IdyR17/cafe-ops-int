# Main Streamlit app

import math
import pandas as pd
import joblib
import streamlit as st

from src.visuals import (
    show_hourly_transaction_volume,
    show_product_demand,
    show_daily_transaction_trend,
)

##### LOAD TRAINED MODELS #####

hourly_model_data = joblib.load("models/hourly_model.joblib")

product_model_data = joblib.load("models/product_model.joblib")


# Separate the trained models from their saved feature columns
hourly_model = hourly_model_data["model"]
hourly_feature_columns = hourly_model_data["feature_columns"]

product_model = product_model_data["model"]
product_feature_columns = product_model_data["feature_columns"]


##### STREAMLIT APP #####

st.title("Café Operations Intelligence")

st.write(
    "Machine learning forecasts for hourly transaction volume "
    "and daily product demand."
)


##### TEST MODEL LOADING #####

st.success("Trained models loaded!")

##### LOAD PRODUCT INFORMATION #####

product_data = pd.read_csv("data/daily_products.csv")

product_list = (
    product_data[
        [
            "product_id",
            "product_detail",
        ]
    ]
    .drop_duplicates()
    .sort_values("product_detail")
)

##### LOAD HOURLY SALES DATA #####

hourly_data = pd.read_csv("data/hourly_sales.csv")

##### HOURLY TRANSACTION FORECAST #####

st.header("Hourly Transaction Forecast")

# Performance per staff member input
productivity_target = st.number_input(
    "Transactions per Staff Member per Hour",
    min_value=1,
    value=10,
    step=1,
)

store_location = st.selectbox(
    "Store Location",
    [
        "Astoria",
        "Hell's Kitchen",
        "Lower Manhattan",
    ],
)

forecast_date = st.date_input(
    "Forecast Date",
)

forecast_hour = st.selectbox(
    "Hour",
    range(6, 21),
)


if st.button("Predict Hourly Transactions"):

    # Get date features from the selected date
    day_of_week = forecast_date.weekday()
    month = forecast_date.month
    is_weekend = int(day_of_week >= 5)

    # Create one row with the same features used to train the model
    prediction_data = pd.DataFrame(
        [
            {
                "store_location": store_location,
                "hour": forecast_hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
            }
        ]
    )

    # Convert store location into number columns
    prediction_data = pd.get_dummies(
        prediction_data,
        columns=["store_location"],
        dtype=int,
    )

    # Make the prediction data match the columns used during training
    prediction_data = prediction_data.reindex(
        columns=hourly_feature_columns,
        fill_value=0,
    )

    # Make the prediction
    predicted_transactions = hourly_model.predict(prediction_data)[0]
    predicted_transactions_rounded = round(predicted_transactions)

    # Estimate staffing using the selected productivity target
    suggested_staff = math.ceil(predicted_transactions / productivity_target)

    st.metric(
        "Predicted Hourly Transactions",
        predicted_transactions_rounded,
    )

    st.metric(
        "Suggested Staffing Level",
        suggested_staff,
    )

    st.caption(
        "Staffing is calculated using the selected transactions-per-staff-member target. "
        "It is an operational recommendation based on predicted demand"
    )

##### PRODUCT DEMAND FORECAST #####

st.header("Daily Product Demand Forecast")


# Create a list that shows both the product ID and product name
product_options = {}

for _, product in product_list.iterrows():
    product_label = str(product["product_id"]) + " - " + product["product_detail"]

    product_options[product_label] = product["product_id"]


product_store_location = st.selectbox(
    "Store Location",
    [
        "Astoria",
        "Hell's Kitchen",
        "Lower Manhattan",
    ],
    key="product_store",
)

product_forecast_date = st.date_input(
    "Forecast Date",
    key="product_date",
)

selected_product = st.selectbox(
    "Product",
    product_options.keys(),
)


if st.button("Predict Product Demand"):

    # Get the selected product ID
    selected_product_id = product_options[selected_product]

    # Get date features from the selected date
    product_day_of_week = product_forecast_date.weekday()
    product_month = product_forecast_date.month
    product_is_weekend = int(product_day_of_week >= 5)

    # Create one row with the same features used to train the model
    product_prediction_data = pd.DataFrame(
        [
            {
                "store_location": product_store_location,
                "product_id": selected_product_id,
                "day_of_week": product_day_of_week,
                "month": product_month,
                "is_weekend": product_is_weekend,
            }
        ]
    )

    # Convert store location and product ID into numeric columns
    product_prediction_data = pd.get_dummies(
        product_prediction_data,
        columns=[
            "store_location",
            "product_id",
        ],
        dtype=int,
    )

    # Make the prediction data match the columns used during training
    product_prediction_data = product_prediction_data.reindex(
        columns=product_feature_columns,
        fill_value=0,
    )

    # Make the prediction
    predicted_units = product_model.predict(product_prediction_data)[0]

    predicted_units_rounded = round(predicted_units)

    # Add a 15% safety buffer for the suggested par level
    suggested_par = math.ceil(predicted_units * 1.15)

    st.metric(
        "Predicted Units Sold",
        predicted_units_rounded,
    )

    st.metric(
        "Suggested Par Level",
        suggested_par,
    )

    st.caption("Suggested par includes a 15% safety buffer above predicted demand.")

##### VISUALIZATIONS #####

show_hourly_transaction_volume(hourly_data)
show_product_demand(product_data)
show_daily_transaction_trend(hourly_data)
