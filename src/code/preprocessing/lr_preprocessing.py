import numpy as np

from .common import create_preprocessor


class LRFeaturePreprocessor:
    # Bentuk fitur tambahan sebelum preprocessing Logistic Regression.

    @staticmethod
    def _augment(X):
        out = X.copy()
        out["year_month"] = (
            out["arrival_year"].astype(str) + "-" + out["arrival_month"].astype(str).str.zfill(2)
        )
        out["special_requests_isna"] = out["special_requests"].isna().astype(float)
        out["special_requests_cat"] = (
            out["special_requests"].fillna(-1).clip(upper=3).astype(int).astype(str)
        )
        out["rate_is_zero"] = (out["nightly_rate_idr"] == 0).astype(float)
        out["total_nights_zero"] = (out["weekend_nights"].eq(0) & out["weekday_nights"].eq(0)).astype(float)
        out["log_dba"] = np.log1p(out["days_before_arrival"])
        out["log_prior_comp"] = np.log1p(out["prior_completed_bookings"])
        out["log_prior_canc"] = np.log1p(out["prior_cancellations"])
        out["has_prior_comp"] = (out["prior_completed_bookings"] > 0).astype(float)
        out["ota_no_req"] = (
            (out["booking_channel"] == "Online Travel Agent")
            & (out["special_requests"].fillna(0) == 0)
        ).astype(float)
        out["dba_gt_150"] = (out["days_before_arrival"] > 150).astype(float)
        sr_bucket = np.where(
            out["special_requests"].isna(), "NA",
            np.where(out["special_requests"] == 0, "0", "pos")
        )
        out["ch_x_sr"] = out["booking_channel"].astype(str) + "_" + sr_bucket
        return out

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        augmented = self._augment(X[self.columns_])
        self.transformer_ = create_preprocessor(augmented)
        self.transformer_.fit(augmented, y)
        return self

    def transform(self, X):
        if not hasattr(self, "transformer_"):
            raise ValueError("Fit the preprocessor before transform")
        return self.transformer_.transform(self._augment(X[self.columns_]))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
