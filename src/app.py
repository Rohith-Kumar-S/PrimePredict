import streamlit as st
import datetime
from main import forcast
import numpy as np
import locale

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

st.title("Amazon Sales Forecast")


start = datetime.date(2023, 1, 1)
temp = datetime.date(2023, 12, 31)
end = datetime.date(2025, 12, 31)

date_range = st.date_input(
    "Select the range to forcast",
    value=(start, temp),
    min_value=start,
    max_value=end,
    format="MM/DD/YYYY",
)


inflation_rate = st.slider("Expected Inflation rate", 0.00, 10.00)

btn = st.button("forcast")

if btn:
    start_date = f"{date_range[0].year}-{date_range[0].month}-{date_range[0].day}"
    end_date = f"{date_range[1].year}-{date_range[1].month}-{date_range[1].day}"

    prediction = forcast(start_date, end_date, inflation_rate)
    prediction.set_index("dates", inplace=True)

    col1, col2 = st.columns(2)
    col1.write("Total Sales")
    col2.write("Average Sales")

    col1, col2 = st.columns(2)
    col1.subheader(f"{locale.currency(int(np.sum(prediction['xgb predictions'])), grouping=True)}")
    col2.subheader(f"{locale.currency(int(np.mean(prediction['xgb predictions'])), grouping=True)}")

    st.line_chart(prediction)
