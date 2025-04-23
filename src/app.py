import streamlit as st
import datetime
from main import forcast, get_state_and_categories_by_frequency
import numpy as np
import locale
from data.dataloader import DataLoader

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

st.set_page_config(
    page_title="PrimeForecast",  
    page_icon="🌟"  
)

if "btn1_clicked" not in st.session_state:
    st.session_state.btn1_clicked = False
    data = DataLoader(is_training=True)
    st.session_state.data_training = data

if "btn2_clicked" not in st.session_state:
    st.session_state.btn2_clicked = False

if "btn3_clicked" not in st.session_state:
    st.session_state.btn3_clicked = False

st.title("Amazon Sales Forecast")


start = datetime.date(2023, 1, 1)
temp = datetime.date(2023, 12, 31)
end = datetime.date(2025, 12, 31)
date_range = []
date_range = st.date_input(
    "Select the range to forcast",
    value=(start, temp),
    min_value=start,
    max_value=end,
    format="MM/DD/YYYY",
)
start_date = None
end_date = None
if len(date_range) == 2:
    start_date = f"{date_range[0].year}-{date_range[0].month}-{date_range[0].day}"
    end_date = f"{date_range[1].year}-{date_range[1].month}-{date_range[1].day}"
# Button 1 logic
if st.button("Forecast Overall Sales"):
    st.session_state.btn1_clicked = True


if st.session_state.btn1_clicked:
    prediction_df, previous_sales, years = forcast(
        start_date, end_date, st.session_state.data_training
    )
    if prediction_df is not None:
        print(previous_sales)
        prediction_df.set_index("dates", inplace=True)
        col1, col2 = st.columns(2)
        col1.write("Total Sales")
        year_string = f"{'Year' if years==0 else 'Years'}"
        col2.write(f"Last {years+1} {year_string} Sales")

        col1, col2 = st.columns(2)
        col1.subheader(
            f"{locale.currency(int(np.sum(prediction_df['Sales Prediction - xbg'])), grouping=True)}"
        )
        col2.subheader(f"{locale.currency(int(np.sum(previous_sales)), grouping=True)}")

        st.line_chart(prediction_df)


st.header("Forcast by State")
states, category = get_state_and_categories_by_frequency(st.session_state.data_training)
us_state_selected = st.selectbox(
    "Select a state **ordered by purchase frequency descending",
    states,
)

# Button 2 logic
if st.button("Forecast State Sales"):
    st.session_state.btn2_clicked = True

if st.session_state.btn2_clicked:
    print(us_state_selected)
    prediction_df, previous_sales, years = forcast(
        start_date,
        end_date,
        st.session_state.data_training,
        us_state_selected,
        is_state=True,
    )
    if prediction_df is not None:
        print(previous_sales)
        prediction_df.set_index("dates", inplace=True)
        col1, col2 = st.columns(2)
        col1.write("Total Sales")
        year_string = f"{'Year' if years==0 else 'Years'}"
        col2.write(f"Last {years+1} {year_string} Sales")

        col1, col2 = st.columns(2)
        col1.subheader(
            f"{locale.currency(int(np.sum(prediction_df['Sales Prediction - xbg'])), grouping=True)}"
        )
        col2.subheader(f"{locale.currency(int(np.sum(previous_sales)), grouping=True)}")

        st.line_chart(prediction_df)


st.header("Forcast by Product Category")
category_selected = st.selectbox(
    "Select a product category **ordered by purchase frequency descending",
    category,
)

if st.button("Forecast Category Sales"):
    st.session_state.btn3_clicked = True

if st.session_state.btn3_clicked:
    print(category_selected)
    prediction_df, previous_sales, years = forcast(
        start_date,
        end_date,
        st.session_state.data_training,
        category_selected,
        is_state=False,
    )
    if prediction_df is not None:
        print(previous_sales)
        prediction_df.set_index("dates", inplace=True)
        col1, col2 = st.columns(2)
        col1.write("Total Sales")
        year_string = f"{'Year' if years==0 else 'Years'}"
        col2.write(f"Last {years+1} {year_string} Sales")

        col1, col2 = st.columns(2)
        col1.subheader(
            f"{locale.currency(int(np.sum(prediction_df['Sales Prediction - xbg'])), grouping=True)}"
        )
        col2.subheader(f"{locale.currency(int(np.sum(previous_sales)), grouping=True)}")

        st.line_chart(prediction_df)
