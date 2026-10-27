**Cafe Operations Intelligence**

Café Operations Intelligence is a machine learning project built for my WGU Computer Science Capstone.

The application uses historical coffee shop sales data to make two types of forecasts:

- Hourly transaction volume to help with staffing decisions
- Daily product demand to help with par and stocking decisions

The goal is not to automatically make operational decisions. The forecasts are meant to give managers another data point to use along with their own judgment.

## Live Application

https://cafeoperations.streamlit.app/

## Dataset

This project uses the public **Maven Analytics Coffee Shop Sales** dataset.

The dataset contains transaction data from three fictitious New York City coffee shop locations from January through June 2023.

No customer, employee, financial, or proprietary sales data is used in this project.

## Machine Learning Models

The project uses two Random Forest regression models built with scikit-learn.

### Hourly Transaction Model

Predicts the number of transactions expected for a selected:

- Store
- Date
- Hour

Features used:

- Store location
- Hour
- Day of week
- Month
- Weekend status

The application also calculates a suggested staffing level using:

`predicted transactions / transactions per staff member`

The result is rounded up to the next whole employee.

### Product Demand Model

Predicts the number of units of a selected product expected to sell during a day.

Features used:

- Store location
- Product ID
- Day of week
- Month
- Weekend status

The application also calculates a suggested par level by adding a 15% safety buffer to the prediction.

## Model Evaluation

The models were trained using January through May 2023 data and tested on June 2023 data.

Mean Absolute Error (MAE) was used as the main evaluation metric.

### Hourly Model

- Historical average baseline MAE: **13.10**
- Random Forest MAE: **11.13**

### Product Model

- Historical average baseline MAE: **3.57**
- Random Forest MAE: **3.17**
- MAE on product-days with 10+ units sold: **4.71**

Both Random Forest models performed better than their historical-average baselines.

The final models use:

- `n_estimators=100`
- `random_state=42`

I also tested additional features such as monthly growth, previous-month trends, recent seven-day product sales, and 200 trees. These did not improve the results enough to justify adding more complexity to the final models.

## Application

The Streamlit application contains three tabs:

### Hourly Transactions

Allows the user to select a store, date, hour, and transactions-per-staff-member target.

The application returns:

- Predicted hourly transactions
- Suggested staffing level

### Product Demand

Allows the user to select a store, date, and product.

The application returns:

- Predicted units sold
- Suggested par level

### Visualizations

Includes three visualizations:

- Average Hourly Transaction Volume
- Average Daily Product Demand
- Daily Transaction Trend

## Project Structure

```text
cafe-ops-int/
├── app.py
├── train.py
├── requirements.txt
├── data/
│   ├── sales.xlsx
│   ├── hourly_sales.csv
│   └── daily_products.csv
├── models/
│   ├── hourly_model.joblib
│   └── product_model.joblib
└── src/
    ├── __init__.py
    ├── data.py
    ├── model.py
    ├── testing.py
    └── visuals.py
```

## Running the Project Locally

### 1. Clone the repository

```powershell
git clone https://github.com/IdyR17/cafe-ops-int.git
cd cafe-ops-int
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.venv\\Scripts\\Activate.ps1
```

### 4. Install the required libraries

```powershell
pip install -r requirements.txt
```

### 5. Train the models

```powershell
python train.py
```

This creates the processed CSV files and saves the trained models in the `models` folder.

### 6. Start the Streamlit application

```powershell
streamlit run app.py
```

## Required Libraries

The project uses:

- Streamlit
- Pandas
- scikit-learn
- Joblib
- OpenPyXL

Exact versions are listed in `requirements.txt`.

## Optional Experiments

The bottom of `train.py` contains commented calls for optional model experiments.

These include:

- Historical monthly growth adjustment
- Previous-month trend features
- Previous seven-day product sales feature

They can be uncommented one at a time and tested by running:

```powershell
python train.py
```

## Data Testing

Data validation and debugging checks are located in:

```text
src/testing.py
```

They can be run as a module when needed.

```powershell
python -m src.testing
```

## Limitations

The source dataset only contains six months of sales data, so the models do not have a full year of seasonality to learn from.

The dataset also does not include:

- Employee schedules or labor hours
- Current inventory

Because of these limitations, the staffing and par outputs should be treated as planning recommendations rather than automatic decisions.

## Author

Jurhidy Rodriguez Tavarez  
WGU Computer Science Capstone
"""
