from src.data import (
    get_data,
    get_hourly_sales,
    get_daily_products,
    fill_missing_product_days,
)


# Print function to test raw, hourly, and product daily sales data
def test_data(sales_data, hourly_data, product_daily_data):
    print(
        "\nFirst 5 sales rows:\n",
        sales_data.head(),
        "\n\nColumns:\n",
        sales_data.columns.tolist(),
        "\n\nNumber of sales rows:",
        len(sales_data),
        "\nUnique transaction IDs:",
        sales_data["transaction_id"].nunique(),
        "\nDate range:",
        sales_data["transaction_date"].min(),
        "to",
        sales_data["transaction_date"].max(),
        "\nStore locations:",
        sales_data["store_location"].unique().tolist(),
        "\nHours in dataset:",
        sorted(sales_data["hour"].unique().tolist()),
        "\n\nFirst 10 hourly rows:\n",
        hourly_data.head(10),
        "\n\nNumber of hourly rows:",
        len(hourly_data),
        "\n\nZero-transaction hourly rows:",
        len(hourly_data[hourly_data["transaction_count"] == 0]),
        "\n\nExample zero-transaction hours:\n",
        hourly_data[hourly_data["transaction_count"] == 0].head(10),
        "\n\nFirst 10 daily product rows:\n",
        product_daily_data.head(10),
        "\n\nNumber of daily product rows:",
        len(product_daily_data),
        "\n\nNumber of unique products:",
        product_daily_data["product_id"].nunique(),
        "\n\nProducts by store:\n",
        product_daily_data.groupby("store_location")["product_id"].nunique(),
        "\n\nDuplicate hourly rows:",
        hourly_data.duplicated(
            subset=["store_location", "transaction_date", "hour"]
        ).sum(),
        "\n\nDuplicate product daily rows:",
        product_daily_data.duplicated(
            subset=["store_location", "transaction_date", "product_id"]
        ).sum(),
    )

    print("\nData info:")
    sales_data.info()

    print("\nHours by store:")
    print(
        sales_data.groupby("store_location")["hour"].agg(["min", "max"]),
        "\n\nZero-unit product days:",
        len(product_daily_data[product_daily_data["units_sold"] == 0]),
        "\n\nExample zero-unit product days:\n",
        product_daily_data[product_daily_data["units_sold"] == 0].head(10),
    )


# Run data testing
def run_data_tests():
    sales_data = get_data()

    hourly_data = get_hourly_sales(sales_data)

    product_daily_data = get_daily_products(sales_data)

    product_daily_data = fill_missing_product_days(
        product_daily_data,
        sales_data,
    )

    test_data(
        sales_data,
        hourly_data,
        product_daily_data,
    )


##### UNCOMMENT TO TEST DATA #####

# run_data_tests()
