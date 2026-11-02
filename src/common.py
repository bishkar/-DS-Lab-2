from typing import Annotated, Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, validate_call
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

OutlierMethod = Literal['iqr', 'zscore']
Contamination = Annotated[float, Field(gt=0, le=0.5)]

DEFAULT_K: Dict[str, float] = {'iqr': 1.5, 'zscore': 3.0}
SCALERS = {'standard': StandardScaler, 'robust': RobustScaler, 'minmax': MinMaxScaler}
SUMMARY_STATS = ['mean', 'std', 'min', 'max']

validated = validate_call(config=ConfigDict(arbitrary_types_allowed=True))


class Schema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class Bounds(Schema):
    lower: float
    upper: float


class MissingSummary(Schema):
    counts: Dict[str, int]
    percentages: Dict[str, float]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def nonzero_counts(self) -> Dict[str, int]:
        return {col: n for col, n in self.counts.items() if n > 0}

    @property
    def nonzero_percentages(self) -> Dict[str, float]:
        return {col: self.percentages[col] for col in self.nonzero_counts}


def numeric_columns(df: pd.DataFrame, exclude: Iterable[Any] = ()) -> List[str]:
    excluded = set(exclude)
    return [col for col in df.select_dtypes(include=[np.number]).columns if col not in excluded]


def present(df: pd.DataFrame, columns: Iterable[Any], numeric: bool = False) -> List[Any]:
    return [col for col in columns
            if col in df.columns and (not numeric or pd.api.types.is_numeric_dtype(df[col]))]


def missing_summary(df: pd.DataFrame) -> MissingSummary:
    counts = df.isnull().sum()
    n_rows = len(df)
    return MissingSummary(
        counts={str(col): int(n) for col, n in counts.items()},
        percentages={str(col): (n / n_rows * 100 if n_rows else 0.0) for col, n in counts.items()},
    )


def count_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum()) if len(df) else 0


def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024 ** 2)


def dtype_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(col): str(dtype) for col, dtype in df.dtypes.items()}


def outlier_bounds(series: pd.Series, method: OutlierMethod, k: float) -> Optional[Bounds]:
    clean = series.dropna()
    if clean.empty:
        return None
    if method == 'iqr':
        q1, q3 = clean.quantile([0.25, 0.75])
        return Bounds(lower=q1 - k * (q3 - q1), upper=q3 + k * (q3 - q1))
    std = clean.std()
    if pd.isna(std) or std == 0:
        return None
    return Bounds(lower=clean.mean() - k * std, upper=clean.mean() + k * std)


def outside(series: pd.Series, bounds: Optional[Bounds]) -> pd.Series:
    if bounds is None:
        return pd.Series(False, index=series.index)
    return (series < bounds.lower) | (series > bounds.upper)


def clip_to_bounds(series: pd.Series, bounds: Optional[Bounds]) -> pd.Series:
    return series if bounds is None else series.clip(lower=bounds.lower, upper=bounds.upper)


def shift_non_negative(series: pd.Series) -> pd.Series:
    return series - min(series.min(), 0)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator.replace(0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0)


def median_filled(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median()).fillna(0)


def fill_mode(df: pd.DataFrame, columns: Iterable[Any]) -> None:
    for col in columns:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])


def lookup_dataset(name: str, original: pd.DataFrame,
                   variants: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    datasets = {'original': original, **variants}
    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(datasets)}")
    return datasets[name]


def stats_by_frame(frames: Dict[str, pd.DataFrame], columns: Optional[List[Any]] = None) -> pd.DataFrame:
    return pd.concat(
        {name: (df if columns is None else df[columns]).describe().loc[SUMMARY_STATS]
         for name, df in frames.items()},
        axis=1,
    )
