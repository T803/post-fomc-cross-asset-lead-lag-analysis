"""Generates input data for future analyses"""

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday, Holiday
import numpy as np
from pathlib import Path
import dukascopy_python

Path(__file__).resolve().parent / "fomc_calendar_data"


def fetch_tick_data(start_date, end_date, instrument):
    """Gets tick data for given instrument, returns 1s forward filled mids"""
    tick_data = dukascopy_python.fetch(
        instrument=instrument,
        interval=dukascopy_python.INTERVAL_TICK,
        offer_side=dukascopy_python.OFFER_SIDE_BID,
        start=start_date,
        end=end_date,
    )

    mid_data = (tick_data["bidPrice"] + tick_data["askPrice"]) / 2
    mid_close = mid_data.to_frame(name=str(instrument))

    return mid_close.resample("1s").last()


def fetch_spreads(start_date, end_date, instrument):
    """Returns 1s-resampled bid, ask, mid, and spread."""
    tick_data = dukascopy_python.fetch(
        instrument=instrument,
        interval=dukascopy_python.INTERVAL_TICK,
        offer_side=dukascopy_python.OFFER_SIDE_BID,
        start=start_date,
        end=end_date,
    )
    out = pd.DataFrame({"bid": tick_data["bidPrice"], "ask": tick_data["askPrice"]})
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["spread"] = out["ask"] - out["bid"]
    return out.resample("1s").last()


def get_fomc_calendar(
    fomc_calendar_csv_path=Path(__file__).resolve().parent
    / "fomc_calendar_data/fomc_dates.csv",
):
    """Gets fomc calendar dataframe with utc dates and times"""
    fomc_calendar_data = pd.read_csv(fomc_calendar_csv_path)
    fomc_calendar_data.set_index("Timestamp", inplace=True)
    fomc_calendar_data.index = pd.to_datetime(
        fomc_calendar_data.index, format="%m/%d/%y %H:%M"
    )
    # We remove the timezone info to properly convert to UTC
    # and avoid issues related to daylight savings time
    naive_dates = fomc_calendar_data.index.tz_localize(None)

    # Forcing the time to be exactly 2:00 PM (when the decision is released), for every event
    fixed_naive_dates = pd.to_datetime(naive_dates.date) + pd.Timedelta(hours=14)
    # Localizing to EST
    ny_times = fixed_naive_dates.tz_localize("America/New_York")
    # Converting to pure UTC for the Dukascopy API
    correct_utc_times = ny_times.tz_convert("UTC")
    # Overwriting the index with the correct utc timestamps
    fomc_calendar_data.index = correct_utc_times
    return fomc_calendar_data


def get_us_market_holidays(start, end):
    """Helper function that gets US market holidays"""

    class _MarketHolidays(USFederalHolidayCalendar):
        rules = USFederalHolidayCalendar.rules + [
            GoodFriday,
            Holiday("Christmas Eve", month=12, day=24),
            Holiday("New Year's Eve", month=12, day=31),
        ]

    return pd.DatetimeIndex(
        _MarketHolidays().holidays(start=start, end=end)
    ).normalize()


def build_control_calendar(
    fomc_calendar_data,
    n_controls_per_event=3,
    exclusion_buffer_days=1,
    max_days_from_event=10,
):
    """For each FOMC meeting, we randomly pick n_controls_per_event trading days within
    max_days_from_event days of the meeting, skipping the meeting day itself, the
    day(s) immediately before/after (exclusion_buffer_days), weekends, and US market
    holidays. Each control day is timestamped at the same time of day as its
    matched FOMC meeting, so it can be pulled with the same event window"""
    fomc_timestamps = fomc_calendar_data.index.tolist()
    us_market_holidays = set(
        get_us_market_holidays(
            fomc_calendar_data.index.min() - pd.Timedelta(days=max_days_from_event + 5),
            fomc_calendar_data.index.max() + pd.Timedelta(days=max_days_from_event + 5),
        ).date
    )
    random_generator = np.random.default_rng(seed=42)

    control_group_dates = []
    matched_fomc_dates = []
    for fomc_timestamp in fomc_timestamps:
        start_window = fomc_timestamp - pd.Timedelta(days=max_days_from_event)
        end_window = fomc_timestamp + pd.Timedelta(days=max_days_from_event)
        possible_dates = pd.date_range(
            start=start_window, end=end_window, freq="D"
        ).tolist()

        # Removing the FOMC day itself and any day within exclusion_buffer_days of it,
        # + weekends (markets are closed)
        possible_dates = [
            day
            for day in possible_dates
            if abs((day - fomc_timestamp).days) > exclusion_buffer_days
            and day.weekday() < 5
        ]

        valid_days = 0
        while valid_days < n_controls_per_event:
            candidate_day = random_generator.choice(possible_dates)
            if candidate_day.date() not in us_market_holidays:
                valid_days += 1
                possible_dates.remove(candidate_day)
                control_group_dates.append(candidate_day)
                matched_fomc_dates.append(fomc_timestamp)

    control_calendar = (
        pd.DataFrame(
            {
                "Timestamp": control_group_dates,
                "matched_fomc_date": matched_fomc_dates,
            }
        )
        .set_index("Timestamp")
        .sort_index()
    )

    return control_calendar
