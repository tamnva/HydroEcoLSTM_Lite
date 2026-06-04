from typing import Iterable

import pandas as pd


class Scaler:
    """Scale selected DataFrame columns using min-max, z-score, or no scaling."""

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    NONE = "none"

    VALID_SCALERS = {MIN_MAX, Z_SCORE, NONE}

    def __init__(self, exclude_columns: Iterable[str] = ("id", "time")) -> None:
        self.exclude_columns = list(exclude_columns)
        self.columns = None
        self.offset = None
        self.scale = None
        self.scaler_names = None

    def fit(self, data: pd.DataFrame, scaler_names: list[str]) -> "Scaler":
        """Calculate scaling parameters for each selected column."""
        self._validate_dataframe(data)

        self.columns = data.columns.drop(self.exclude_columns, errors="ignore")

        if len(scaler_names) != len(self.columns):
            raise ValueError(
                f"Expected {len(self.columns)} scaler names, "
                f"but got {len(scaler_names)}."
            )

        invalid_scalers = set(scaler_names) - self.VALID_SCALERS
        if invalid_scalers:
            raise ValueError(f"Invalid scaler name(s): {sorted(invalid_scalers)}")

        stats = data[self.columns].agg(["min", "max", "mean", "std"])

        offset = pd.Series(0.0, index=self.columns)
        scale = pd.Series(1.0, index=self.columns)

        for column, scaler_name in zip(self.columns, scaler_names):
            if scaler_name == self.MIN_MAX:
                column_range = stats.loc["max", column] - stats.loc["min", column]
                if column_range != 0:
                    offset[column] = stats.loc["min", column]
                    scale[column] = column_range

            elif scaler_name == self.Z_SCORE:
                column_std = stats.loc["std", column]
                if column_std != 0:
                    offset[column] = stats.loc["mean", column]
                    scale[column] = column_std

        self.offset = offset
        self.scale = scale
        self.scaler_names = pd.Series(scaler_names, index=self.columns)

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale the data using the fitted parameters."""
        self._check_is_fitted()
        self._validate_dataframe(data)

        scaled_data = data.copy()
        columns = self.columns.intersection(scaled_data.columns)

        scaled_data[columns] = (
            scaled_data[columns] - self.offset[columns]
        ) / self.scale[columns]

        scaled_data[columns] = scaled_data[columns].astype("float32")
        
        return scaled_data

    def inverse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reverse the scaling operation."""
        self._check_is_fitted()
        self._validate_dataframe(data)

        original_data = data.copy()
        columns = self.columns.intersection(original_data.columns)

        original_data[columns] = (
            self.offset[columns] + self.scale[columns] * original_data[columns]
        )

        original_data[columns] = original_data[columns].astype("float32")
        
        return original_data

    def fit_transform(self, data: pd.DataFrame, scaler_names: list[str]) -> pd.DataFrame:
        """Fit the scaler and return the scaled data."""
        return self.fit(data, scaler_names).transform(data)

    @staticmethod
    def _validate_dataframe(data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

    def _check_is_fitted(self) -> None:
        if self.columns is None or self.offset is None or self.scale is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")