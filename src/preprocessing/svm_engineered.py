"""Optional, deterministic booking features for the SVM preprocessing pipeline.

This transformer only uses attributes available at prediction time. Missing
values are preserved for the downstream imputer to learn from training data.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BookingFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, dates=True, durations=True, party=True, missing=False, log=False,
                 missing_columns=None):
        self.dates = dates
        self.durations = durations
        self.party = party
        self.missing = missing
        self.log = log
        self.missing_columns = missing_columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        result = X.copy()
        if self.missing:
            for col in self.feature_names_in_:
                if col in (self.missing_columns or (
                    "num_adults", "num_children", "meal_package", "needs_parking",
                    "room_category", "days_before_arrival", "nightly_rate_idr", "special_requests"
                )):
                    result[f"{col}__missing"] = result[col].isna().astype(float)
        if self.dates:
            arrival = pd.to_datetime(dict(year=result.arrival_year,
                                          month=result.arrival_month,
                                          day=result.arrival_day), errors="coerce")
            booking = arrival - pd.to_timedelta(result.days_before_arrival, unit="D")
            result["arrival_day_of_week"] = arrival.dt.dayofweek.astype(float)
            result["arrival_day_of_year_sin"] = np.sin(2 * np.pi * arrival.dt.dayofyear / 365.25)
            result["arrival_day_of_year_cos"] = np.cos(2 * np.pi * arrival.dt.dayofyear / 365.25)
            result["booking_day_of_week"] = booking.dt.dayofweek.astype(float)
            result["booking_day_of_year_sin"] = np.sin(2 * np.pi * booking.dt.dayofyear / 365.25)
            result["booking_day_of_year_cos"] = np.cos(2 * np.pi * booking.dt.dayofyear / 365.25)
        if self.durations:
            result["total_nights"] = result.weekend_nights + result.weekday_nights
            result["estimated_total_price"] = result.total_nights * result.nightly_rate_idr
        if self.party:
            result["total_guests"] = result.num_adults + result.num_children
            result["has_children"] = np.where(result.num_children.isna(), np.nan,
                                                (result.num_children > 0).astype(float))
        if self.log:
            for col in ("days_before_arrival", "nightly_rate_idr", "prior_cancellations", "prior_completed_bookings"):
                result[f"{col}__log"] = np.log1p(result[col].clip(lower=0))
        return result
