# Charts for visualization of data
import streamlit as st

##################################################
##### VISUALIZATIONS #####
##################################################


##### VISUALIZATION 1: HOURLY TRANSACTION VOLUME #####


def show_hourly_transaction_volume(hourly_data):

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

    store_hourly_data = hourly_data[
        hourly_data["store_location"] == selected_chart_store
    ]

    average_hourly_transactions = (
        store_hourly_data.groupby("hour")["transaction_count"].mean().reset_index()
    )

    st.line_chart(
        average_hourly_transactions,
        x="hour",
        y="transaction_count",
    )


##### VISUALIZATION 2: PRODUCT DEMAND #####


def show_product_demand(product_data):

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


def show_daily_transaction_trend(hourly_data):

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
