# Data processing script
import pandas as pd

FILE_PATH = "data/sales.xlsx"


# loading the excel file into a pandas dataframe
def get_data():
    sales_data = pd.read_excel(FILE_PATH)

    # Create an hour column from the transaction_time column
    sales_data["hour"] = [time.hour for time in sales_data["transaction_time"]]

    # Testing data
    print(
        "\nFirst 5 rows:\n",
        sales_data.head(),
        "\n\nColumns:\n",
        sales_data.columns.tolist(),
        "\n\nData info:",
    )
    print(
        "\nNumber of rows:",
        len(sales_data),
        "\nUnique transaction IDs:",
        sales_data["transaction_id"].nunique(),
        "\nDate range:",
        sales_data["transaction_date"].min(),
        "to",
        sales_data["transaction_date"].max(),
        "\nStore locations:",
        sales_data["store_location"].unique().tolist(),
        "\nTransaction time example:",
        sales_data["transaction_time"].iloc[0],
        "\nTransaction time type:",
        type(sales_data["transaction_time"].iloc[0]),
    )

    print("\nHours in dataset:", sorted(sales_data["hour"].unique().tolist()))

    print("\nHours by store:")

    print(sales_data.groupby("store_location")["hour"].agg(["min", "max"]))


get_data()
