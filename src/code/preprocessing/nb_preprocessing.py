import numpy as np
import pandas as pd


class NaiveBayesPreprocessor:
    # Batas diskritisasi dipelajari hanya dari data training.

    def __init__(self, discretize=True, n_bins=15):
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self.discretize = bool(discretize)
        self.n_bins = int(n_bins)

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        if not self.discretize:
            return self
        self.bin_edges_ = {}
        for col, nb in [("days_before_arrival", self.n_bins),
                        ("nightly_rate_idr", self.n_bins),
                        ("arrival_day", 6)]:
            if col not in X.columns:
                continue
            valid = X[col].dropna()
            if col == "nightly_rate_idr":
                valid = valid[valid > 0]
            if len(valid) >= 2:
                self.bin_edges_[col] = np.unique(
                    np.quantile(valid, np.linspace(0, 1, nb + 1)[1:-1])
                )
            else:
                self.bin_edges_[col] = np.array([], dtype=float)
        return self

    def transform(self, X):
        if not hasattr(self, "columns_"):
            raise ValueError("Fit the preprocessor before transform")
        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = X[self.columns_].copy()
        if not self.discretize:
            return df

        out = pd.DataFrame(index=df.index)
        out["year_month"] = (
            df["arrival_year"].astype(str) + "-" + df["arrival_month"].astype(str).str.zfill(2)
        )
        for col in ["meal_package", "room_category", "booking_channel"]:
            out[col] = df[col].fillna("__MISSING__").astype(str)
        for col in ["num_adults", "num_children", "weekend_nights", "needs_parking", "special_requests"]:
            out[col] = df[col].fillna(-1).astype(int).astype(str)

        out["prior_cancel_bin"] = np.where(
            df["prior_cancellations"] == 0, "0",
            np.where(df["prior_cancellations"] == 1, "1", "2+")
        )
        out["prior_comp_bin"] = np.where(
            df["prior_completed_bookings"] == 0, "0",
            np.where(df["prior_completed_bookings"] <= 2, "1-2", "3+")
        )
        out["weekday_nights_bin"] = df["weekday_nights"].clip(upper=7).astype(int).astype(str)
        out["rate_is_zero"] = np.where(
            df["nightly_rate_idr"].isna(), "NA",
            np.where(df["nightly_rate_idr"] == 0, "0", "pos")
        )
        out["total_nights_zero"] = ((df["weekend_nights"] + df["weekday_nights"]) == 0).astype(str)

        for col, edges in self.bin_edges_.items():
            v = df[col]
            binned = np.where(
                v.isna(), "NA",
                np.searchsorted(edges, v.fillna(0.0), side="right").astype(str)
            )
            if col == "nightly_rate_idr":
                binned = np.where(v == 0, "ZERO", binned)
            out[f"{col}_bin"] = binned
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
