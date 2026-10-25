# Main Streamlit app
import pandas as pd
import joblib
import streamlit as st

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

    st.metric(
        "Predicted Transactions",
        round(predicted_transactions),
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

    st.metric(
        "Predicted Units Sold",
        round(predicted_units),
    )

##### VISUALIZATION 1: HOURLY TRANSACTION VOLUME #####

st.header("Average Hourly Transaction Volume")

selected_chart_store = st.selectbox(
    "Select Store for Hourly Chart",
    [
        "Astoria",
        "Hell's Kitchen",
        "Lower Manhattan",
    ],
    key="hourly_chart_store",
)

store_hourly_data = hourly_data[hourly_data["store_location"] == selected_chart_store]

average_hourly_transactions = (
    store_hourly_data.groupby("hour")["transaction_count"].mean().reset_index()
)

st.line_chart(
    average_hourly_transactions,
    x="hour",
    y="transaction_count",
)

##### VISUALIZATION 2: PRODUCT DEMAND #####

st.header("Average Daily Product Demand")

selected_product_store = st.selectbox(
    "Select Store for Product Chart",
    [
        "Astoria",
        "Hell's Kitchen",
        "Lower Manhattan",
    ],
    key="product_chart_store",
)

store_product_data = product_data[
    product_data["store_location"] == selected_product_store
]

average_product_demand = (
    store_product_data.groupby(["product_id", "product_detail"])["units_sold"]
    .mean()
    .reset_index()
    .sort_values(
        "units_sold",
        ascending=False,
    )
    .head(10)
)

st.bar_chart(
    average_product_demand,
    x="product_detail",
    y="units_sold",
)

##### VISUALIZATION 3: DAILY TRANSACTION TREND #####

st.header("Daily Transaction Trend")

selected_daily_store = st.selectbox(
    "Select Store for Daily Trend",
    [
        "Astoria",
        "Hell's Kitchen",
        "Lower Manhattan",
    ],
    key="daily_trend_store",
)

store_daily_data = hourly_data[
    hourly_data["store_location"] == selected_daily_store
].copy()

daily_transactions = (
    store_daily_data.groupby("transaction_date")["transaction_count"]
    .sum()
    .reset_index()
)

st.area_chart(
    daily_transactions,
    x="transaction_date",
    y="transaction_count",
)
