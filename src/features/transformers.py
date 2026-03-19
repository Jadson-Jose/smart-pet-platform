import pandas as pd
import numpy as np


class CastrationTransformer:
    """Extrai status de castração de 'Sex upon Outcome'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def parse(val):
            if pd.isna(val):
                return 'Unknown'
            val = str(val).lower()
            if 'neutered' in val or 'spayed' in val:
                return 'Castrado'
            if 'intact' in val:
                return 'Inteiro'
            return 'Unknown'

        return X.apply(parse)


class AgeTransformer:
    """Converte 'Age upon Outcome' para dias numéricos."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def parse(val):
            if pd.isna(val):
                return np.nan
            val = str(val).lower().strip()
            try:
                num = int(val.split()[0])
                if 'year'  in val: return num * 365
                if 'month' in val: return num * 30
                if 'week'  in val: return num * 7
                if 'day'   in val: return num
            except Exception:
                return np.nan
            return np.nan

        return X.apply(parse)


class HasNameTransformer:
    """Cria feature binária: animal tem nome? 1 = sim, 0 = não."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.notna().astype(int)


class MonthCyclicTransformer:
    """Extrai mês do DateTime e aplica transformação cíclica."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        months = pd.to_datetime(X, format='mixed', utc=True).dt.month
        month_sin = np.sin(2 * np.pi * months / 12)
        month_cos = np.cos(2 * np.pi * months / 12)
        return pd.DataFrame({
            'month_sin': month_sin,
            'month_cos': month_cos
        })


class ColorTransformer:
    """Extrai a cor primária de 'Color'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna('Unknown').apply(
            lambda c: str(c).split('/')[0].strip()
        )


class BreedTransformer:
    """Simplifica raças com menos de min_count ocorrências para 'Mixed/Other'."""

    def __init__(self, min_count=100):
        self.min_count = min_count
        self.valid_breeds = None

    def fit(self, X, y=None):
        counts = X.value_counts()
        self.valid_breeds = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X):
        return X.apply(
            lambda b: b if b in self.valid_breeds else 'Mixed/Other'
        )