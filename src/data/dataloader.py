import pandas as pd
import os
import kagglehub
import subprocess
import gdown


class DataLoader:

    def __init__(self):
        self.purchases = self.fetch_data(
            "1HdJj68eO9NTZlpwZcLYqdtPqrbKs1cxj", "amazon_purchases.csv"
        )
        self.holidays = self.load_holidays()
        self.inflation = self.fetch_data(
            "1uuTBlEA2caOIyv4-R_G2OlekBCw-i_Le", "inflation.csv"
        )
        self.amazon_events = self.load_amazon_events()
        self.holidays_past_2021 = self.load_holidays_past_2021()

    def purchases(self):
        return self.purchases

    def holidays(self):
        return self.holidays

    def inflation(self):
        return self.inflation

    def amazon_events(self):
        return self.amazon_events

    def holidays_past_2021(self):
        return self.holidays_past_2021
    
    def get_usa_states(self):
        return list(
            self.purchases["Shipping Address State"]
            .dropna()
            .sort_values()
            .unique()
        )

    def download_dataset(self, key, dataset_name):
        path = os.path.abspath(dataset_name)
        if not os.path.exists(path):
            cmd = f"gdown --fuzzy {key} -O {dataset_name}"
            subprocess.run(cmd, shell=True, check=True)
        print(f"Loading>> {dataset_name} path: {path}")
        return path

    def fetch_data(self, key, filename):
        # Load the dataset from the given path
        path = self.download_dataset(key, filename)
        data = pd.read_csv(path)
        return data

    def load_holidays(self):
        # Load the dataset from the given path
        path = kagglehub.dataset_download("donnetew/us-holiday-dates-2004-2021")
        data = pd.read_csv(os.path.join(path, "US Holiday Dates (2004-2021).csv"))
        return data

    def load_amazon_events(self):
        # Load the dataset from the given path
        amazon_events = self.add_events(
            [["2024-03-20", "2024-03-25"], ["2025-03-25", "2025-03-31"]]
        )
        spring_sale_event_df = pd.DataFrame(
            {
                "Event Date": amazon_events,
                "Amazon Events": ["Big Spring Sale"] * len(amazon_events),
            }
        ).set_index("Event Date")

        amazon_events = self.add_events(
            [
                ["2018-07-16", "2018-07-17"],
                ["2019-07-15", "2019-07-16"],
                ["2020-10-13", "2020-10-14"],
                ["2021-06-21", "2021-06-22"],
                ["2022-07-12", "2022-07-13"],
                ["2023-07-11", "2023-07-12"],
                ["2024-07-16", "2024-07-17"],
                ["2025-07-23", "2025-07-24"],
            ]
        )
        prime_day_event_df = pd.DataFrame(
            {
                "Event Date": amazon_events,
                "Amazon Events": ["Amazon Prime Day"] * len(amazon_events),
            }
        ).set_index("Event Date")

        amazon_events = self.add_events(
            [
                ["2022-10-11", "2022-10-12"],  # Prime Early Access Sale
                ["2023-10-10", "2023-10-11"],  # First official Prime Big Deal Days
                ["2024-10-08", "2024-10-09"],
                ["2025-10-14", "2025-10-15"],
            ]
        )
        prime_big_deal_event_df = pd.DataFrame(
            {
                "Event Date": amazon_events,
                "Amazon Events": ["Prime Big Deal Days"] * len(amazon_events),
            }
        ).set_index("Event Date")

        amazon_events = self.add_events(
            [
                ["2018-11-16", "2018-11-23"],
                ["2019-11-22", "2019-11-29"],
                ["2020-11-20", "2020-11-27"],
                ["2021-11-19", "2021-11-26"],
                ["2022-11-24", "2022-11-25"],
                ["2023-11-17", "2023-11-24"],
                ["2024-11-21", "2024-11-29"],
                ["2025-11-28", "2025-12-01"],
            ]
        )
        black_friday_event_df = pd.DataFrame(
            {
                "Event Date": amazon_events,
                "Amazon Events": ["Black Friday"] * len(amazon_events),
            }
        ).set_index("Event Date")

        amazon_events = self.add_events(("2018-12-02", "2018-12-13"), is_repeat=True)
        _12_day_of_deals_event_df = pd.DataFrame(
            {
                "Event Date": amazon_events,
                "Amazon Events": ["12 Days of Deals"] * len(amazon_events),
            }
        ).set_index("Event Date")

        amazon_events = self.add_events(("2018-12-26", "2018-12-31"), is_repeat=True)
        year_end_clearance_event_df = pd.DataFrame(
            {
                "Event Date": amazon_events,
                "Amazon Events": ["Year-End Clearance Sale"] * len(amazon_events),
            }
        ).set_index("Event Date")

        amazon_events_df = pd.concat(
            [
                spring_sale_event_df,
                prime_day_event_df,
                prime_big_deal_event_df,
                black_friday_event_df,
                _12_day_of_deals_event_df,
                year_end_clearance_event_df,
            ],
            axis=0,
        ).sort_index()

        return amazon_events_df

    def load_holidays_past_2021(self):
        _2022 = pd.Series(
            [
                "2022-12-26",
                "2022-11-24",
                "2022-11-11",
                "2022-10-10",
                "2022-09-05",
                "2022-07-04",
                "2022-06-20",
                "2022-05-30",
                "2022-02-21",
                "2022-01-21",
                "2023-01-02",
                "2023-01-16",
                "2023-02-20",
            ]
        )

        _2022 = pd.to_datetime(_2022)
        dates = pd.Series(
            [
                "January 02 2023",
                "January 16 2023",
                "February 20 2023",
                "May 29 2023",
                "June 19 2023",
                "July 04 2023",
                "September 04 2023",
                "October 09 2023",
                "November 10 2023",
                "November 23 2023",
                "December 25 2023",
                "January 01 2024",
                "January 15 2024",
                "February 19 2024",
                "May 27 2024",
                "June 19 2024",
                "July 04 2024",
                "September 02 2024",
                "October 14 2024",
                "November 11 2024",
                "November 28 2024",
                "December 25 2024",
                "January 01 2025",
                "January 20 2025",
                "January 20 2025",
                "February 17 2025",
                "May 26 2025",
                "June 19 2025",
                "July 04 2025",
                "September 01 2025",
                "October 13 2025",
                "November 11 2025",
                "November 27 2025",
                "December 25 2025",
                "January 01 2026",
                "January 19 2026",
                "February 16 2026",
                "May 25 2026",
                "June 19 2026",
                "July 03 2026",
                "September 07 2025",
                "October 12 2026",
                "November 11 2026",
                "November 26 2026",
                "December 25 2026",
            ]
        )
        fedral_holidays_2023_plus = pd.to_datetime(dates)
        fedral_holdidays_22_plus = pd.concat(
            [_2022, fedral_holidays_2023_plus], axis=0
        ).sort_values()
        fedral_holdidays_22_plus = pd.Series(fedral_holdidays_22_plus.unique())
        fedral_holdidays_22_plus.index = range(len(fedral_holdidays_22_plus))
        fedral_holdidays_22_plus = pd.DataFrame(fedral_holdidays_22_plus).rename(
            columns={0: "date"}
        )
        fedral_holdidays_22_plus["holiday"] = True
        fedral_holdidays_22_plus["date"] = pd.to_datetime(
            fedral_holdidays_22_plus["date"]
        )
        fedral_holdidays_22_plus = fedral_holdidays_22_plus.set_index("date")
        return fedral_holdidays_22_plus

    def add_events(self, events_timestamps, is_repeat=False):
        dummy_date = '2017-04-10'
        events = pd.DatetimeIndex([dummy_date])
        if is_repeat:
            for i in range(len([2018,2019,2020,2021,2022,2023,2024,2025,2026])):
                events = events.append(pd.date_range(start=events_timestamps[0], end=events_timestamps[1]) \
                                    + pd.tseries.offsets.DateOffset(months=i*12))
        else:
            for event in events_timestamps:
                events = events.append(pd.date_range(start=event[0], end=event[1]))
        return events[1:]
    
