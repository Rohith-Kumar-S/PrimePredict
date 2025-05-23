# PrimePredict

Summary:

Amazon’s marketplace is a vast, dynamic retail ecosystem generating millions of daily transactions across countless product categories, offering a rich dataset for uncovering real‑world patterns in purchase frequency, seasonality, and product performance. These insights are critical to optimizing inventory, pricing, and marketing strategies. To help businesses set revenue goals, allocate budgets, and predict growth across all U.S. states, our package is organized into four intuitive modules (EDA, Preprocessing, Machine‑Learning Forecasting, and Clustering) that go beyond traditional time‑series tools like Prophet or ARIMA. By combining regression and tree‑ensemble models (XGBoost with a gblinear booster and CatBoost with random forests) with K‑Means clustering, it produces accurate point forecasts alongside actionable customer‑segment insights, all within a streamlined, user‑friendly framework.


Installation:

-- In the root directory, you can find the requirements file. To run the app, the libraries and packages listed in the requirements file must be installed.
Run conda env create -f requirements.yml to set up a conda environment, or
run pip install -r requirements.txt to install the required libraries and packages directly.
-- If you're using a conda environment, activate it. Then, make sure you're in the project directory: PrimePredict/src.
From there, type streamlit run app.py to start the app.
-- Done!

Organization of modules:

<pre><code>
src/
├── app.py
├── primepredict.py
├── data/
    └── tests
        └── test_dataloader.py
│   ├── dataloader.py
│   ├── preprocessed_datasets/
│   └── raw_datasets/
├── eda/
    └── EDA.ipynb
    └── Features_and_Model-Statewise.ipynb
    └── Features_and_Model.ipynb
├── features/
    └── tests
        └── test_featureengineering.py
│   └── featureengineering.py
├── models/
    └── tests
        └── test_forecast.py
│   ├── forecast.py
│   └── saves/
├── preprocessing/
    └── tests
        └── test_datapreprocessing.py
    └── datapreprocessing.py
</code></pre>


Usage:

This app was built with the idea that it could also be used by individual manufacturing companies whose products are sold on Amazon and who are trying to add a new product to an existing category. For manufacturers, it provides a quick overview of the performance of a product category. Since time to market is key to sales, they can decide when to manufacture and manage inventory based on demand. Given the forecast date range, the application enables users to forecast overall total sales, total sales for any specified U.S. state, and total sales for any specified product category. When the user clicks the Forecast button, the app first trains the model based on the user’s selection if it hasn’t already been trained on that specific selection. The trained model is then saved in the "models/saves" directory. **Therefore, if the model has not yet been trained on that specific selection, the user must click Forecast once to train the model and then click Forecast again to retrieve the forecast results**. The Streamlit app, written in "app.py", calls the functions available in "primepredict.py" to populate the results in the application. If the user performs a forecast based on a state or category, the application first loads the data using **DataLoader** and generates a processed dataset and stores it in "data/processed_datasets", which is then used during forecasting. To start the application, install the necessary packages listed in the requirements.txt file and activate the environment. Then, in that environment, navigate to the src folder and run "streamlit run app.py". The application will boot up and be ready to use.

Example:

from primepredict import PrimePredict

from data.dataloader import DataLoader

data = DataLoader(is_training=True)

returns None at first, try again to fetch  overall amazon sales forecast

overall_prediction_df, previous_sales, years = PrimePredict().forcast(
        "2023-04-04", "2024-04-04", data
    )

returns None at first, try again to fetch statewise sales forecast

statewise_prediction_df, previous_sales, years = PrimePredict().forcast(
        "2023-04-04",
        "2024-04-04",
        data,
        "CA",
        is_state=True,
    )

returns None at first, try again to fetch  category wise sales forecast

categorywise_prediction_df, previous_sales, years = PrimePredict().forcast(
        "2023-04-04",
        "2024-04-04",
        data,
        "ABIS_BOOK",
        is_state=False,
    )
