# Data processing script
import pandas as pd

FILE_PATH = "data/sales.xlsx"


# loading the excel file into a pandas dataframe
def get_data():
    sales_data = pd.read_excel(FILE_PATH)

    # Create an hour column from the transaction_time column
    sales_data["hour"] = [time.hour for time in sales_data["transaction_time"]]

    return sales_data


# Function to get hourly sales data grouped by store location, date, and hour
# And how many transactions happened in that hour
def get_hourly_sales(sales_data):
    hourly_data = (
        sales_data.groupby(["store_location", "transaction_date", "hour"])
        .size()  # count the number of transactions in each group
        .reset_index(name="transaction_count")  # group turned into a table
    )
    # Fill in missing hours for each store and date, so that we have a complete hourly dataset for ML training
    # List that will hold the completed hourly rows
    complete_hourly_data = []

    # Go through each store and date separately
    for (store, date), day_data in hourly_data.groupby(
        ["store_location", "transaction_date"]
    ):

        # Find the first and last transaction hour for that day
        first_hour = day_data["hour"].min()
        last_hour = day_data["hour"].max()

        # Create a dictionary of hour and transaction count
        transaction_counts = dict(zip(day_data["hour"], day_data["transaction_count"]))

        # Create every hour between the first and last transaction
        for hour in range(first_hour, last_hour + 1):
            complete_hourly_data.append(
                {
                    "store_location": store,
                    "transaction_date": date,
                    "hour": hour,
                    # Use the transaction count if the hour exists
                    # Otherwise use 0
                    "transaction_count": transaction_counts.get(hour, 0),
                }
            )

    # Turn the completed rows back into a pandas DataFrame
    hourly_data = pd.DataFrame(complete_hourly_data)

    # Create date specifications for the ML model: day of week, month, and whether it's a weekend
    hourly_data["day_of_week"] = hourly_data["transaction_date"].dt.dayofweek
    hourly_data["month"] = hourly_data["transaction_date"].dt.month
    hourly_data["is_weekend"] = hourly_data["day_of_week"].isin([5, 6]).astype(int)

    return hourly_data


# Get daily product sales by store, date, and product
# The units_sold column shows how many items were sold in each category
def get_daily_products(sales_data):
    product_daily_data = (
        sales_data.groupby(
            [
                "store_location",
                "transaction_date",
                "product_id",
                "product_category",
                "product_type",
                "product_detail",
            ]
        )["transaction_qty"]
        .sum()
        .reset_index(name="units_sold")
    )
    return product_daily_data


# Fill in product days with 0 units when the product had no sales
# Same idea as the hourly sales function, but for products instead of hours
# Only use dates between the product's first and last sale at that store
def fill_missing_product_days(product_daily_data, sales_data):
    complete_product_data = []

    # Go through each store and product separately
    for (store, product_id), product_data in product_daily_data.groupby(
        ["store_location", "product_id"]
    ):
        # Find when this product first and last sold at this store
        first_date = product_data["transaction_date"].min()
        last_date = product_data["transaction_date"].max()
        # Get dates when the store had transactions during this product's active period
        active_store_dates = (
            sales_data[
                (sales_data["store_location"] == store)
                & (sales_data["transaction_date"] >= first_date)
                & (sales_data["transaction_date"] <= last_date)
            ]["transaction_date"]
            .drop_duplicates()
            .sort_values()
        )

        # Save the product information
        product_info = product_data.iloc[0]

        # Create a dictionary containing date and units sold
        units_by_date = dict(
            zip(
                product_data["transaction_date"],
                product_data["units_sold"],
            )
        )

        # Create one row for every active store date
        for date in active_store_dates:
            complete_product_data.append(
                {
                    "store_location": store,
                    "transaction_date": date,
                    "product_id": product_id,
                    "product_category": product_info["product_category"],
                    "product_type": product_info["product_type"],
                    "product_detail": product_info["product_detail"],
                    # Use actual units sold if the date exists
                    # Otherwise use 0
                    "units_sold": units_by_date.get(date, 0),
                }
            )

    # Turn the completed rows into a Dataframe
    product_daily_data = pd.DataFrame(complete_product_data)

    # Create date specifications for the product demand ML model
    product_daily_data["day_of_week"] = product_daily_data[
        "transaction_date"
    ].dt.dayofweek
    product_daily_data["month"] = product_daily_data["transaction_date"].dt.month
    product_daily_data["is_weekend"] = (
        product_daily_data["day_of_week"].isin([5, 6]).astype(int)
    )

    return product_daily_data


# Save the processed datasets as CSV files
def save_processed_data(hourly_data, product_daily_data):
    hourly_data.to_csv("data/hourly_sales.csv", index=False)
    product_daily_data.to_csv("data/daily_products.csv", index=False)


# Function to prepare the data for ML training
def prep_data():
    sales_data = get_data()

    hourly_data = get_hourly_sales(sales_data)

    product_daily_data = get_daily_products(sales_data)
    product_daily_data = fill_missing_product_days(product_daily_data, sales_data)

    return hourly_data, product_daily_data


# print function to test raw, hourly, and product daily sales data
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


# # Uncomment for testing data
# sales_data = get_data()
# hourly_data = get_hourly_sales(sales_data)
# product_daily_data = get_daily_products(sales_data)
# product_daily_data = fill_missing_product_days(product_daily_data, sales_data)

# save_processed_data(hourly_data, product_daily_data)
# test_data(sales_data, hourly_data, product_daily_data)
