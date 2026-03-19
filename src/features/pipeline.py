import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.transformers import (
    AgeTransformer,
    BreedTransformer,
    CastrationTransformer,
    ColorTransformer,
    HasNameTransformer,
    MonthCyclicTransformer,
)


def build_feature_pipeline():
    """
    Constrói o pipeline completo de features.
    Retorna um ColumnTransformer pronto para fit/transform.
    """

    # ── 1. Features categóricas — One-Hot Encoding ──────────
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    # ── 2. Feature numérica — idade em dias ─────────────────
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])

    # ── 3. Assemblar tudo num ColumnTransformer ──────────────
    preprocessor = ColumnTransformer(transformers=[
        ('castration', categorical_pipeline, ['castration_status']),
        ('animal_type', categorical_pipeline, ['Animal Type']),
        ('color',       categorical_pipeline, ['primary_color']),
        ('breed',       categorical_pipeline, ['breed_simplified']),
        ('age',         numeric_pipeline,     ['age_in_days']),
        ('has_name',    'passthrough',         ['has_name']),
        ('month_sin',   'passthrough',         ['month_sin']),
        ('month_cos',   'passthrough',         ['month_cos']),
    ])

    return preprocessor


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas as transformações brutas no DataFrame.
    Retorna df com as novas colunas prontas para o pipeline.
    """

    df = df.copy()

    # Castração
    df['castration_status'] = CastrationTransformer().transform(
        df['Sex upon Outcome']
    )

    # Idade em dias
    df['age_in_days'] = AgeTransformer().transform(
        df['Age upon Outcome']
    )

    # Tem nome
    df['has_name'] = HasNameTransformer().transform(df['Name'])

    # Mês cíclico
    month_df = MonthCyclicTransformer().transform(df['DateTime'])
    df['month_sin'] = month_df['month_sin']
    df['month_cos'] = month_df['month_cos']

    # Cor primária
    df['primary_color'] = ColorTransformer().transform(df['Color'])

    # Raça simplificada
    breed_t = BreedTransformer(min_count=100)
    breed_t.fit(df['Breed'])
    df['breed_simplified'] = breed_t.transform(df['Breed'])

    return df