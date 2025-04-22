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

src/
├── app.py
├── main.py
├── data/
│   ├── dataloader.py
│   ├── preprocessed_datasets/
│   └── raw_datasets/
├── eda/
├── features/
│   └── feature_engineering.py
├── models/
│   ├── forecast.py
│   └── saves/
├── preprocessing/
│   └── datapreprocessing.py
└── utils/
    └── utilities.py


