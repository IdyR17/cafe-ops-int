# Data processing script
import pandas as pd

FILE_PATH = "data/sales.xlsx"


# loading the excel file into a pandas dataframe
def get_data():
    sales_data = pd.read_excel(FILE_PATH)

    # Create an hour column from the transaction_time column
    sales_data["hour"] = [time.hour for time in sales_data["transaction_time"]]

    return sales_data


sales_data = get_data()


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

    return hourly_data


# print function to test raw and hourly sales data
def test_data(sales_data, hourly_data):
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
    )

    print("\nData info:")
    sales_data.info()

    print("\nHours by store:")
    print(sales_data.groupby("store_location")["hour"].agg(["min", "max"]))


sales_data = get_data()
hourly_data = get_hourly_sales(sales_data)

# Uncomment for testing data
test_data(sales_data, hourly_data)
