import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.features.pipeline import build_feature_pipeline, prepare_features

# ── Mapeamento de desfechos ──────────────────────────────────
OUTCOME_MAP = {
    "Adoption": "Adoption",
    "Transfer": "Transfer",
    "Return to Owner": "Return to Owner",
    "Euthanasia": "Euthanasia",
    "Died": "Euthanasia",  # agrupar com Euthanasia
    "Rto-Adopt": "Other",
    "Disposal": "Other",
    "Missing": "Other",
    "Relocate": "Other",
    "Stolen": "Other",
    "Lost": "Other",
}


def load_and_prepare(filepath: str):
    """Carrega o CSV, aplica features e prepara X e y."""
    df = pd.read_csv(filepath)

    # Remover linhas sem desfecho
    df = df.dropna(subset=["Outcome Type"])

    # Criar variável alvo
    df["target"] = df["Outcome Type"].map(OUTCOME_MAP)
    df = df.dropna(subset=["target"])

    # Aplicar feature engineering
    df = prepare_features(df)

    # Separar X e y
    feature_cols = [
        "castration_status",
        "Animal Type",
        "primary_color",
        "breed_simplified",
        "age_in_days",
        "has_name",
        "month_sin",
        "month_cos",
    ]
    X = df[feature_cols]
    y = df["target"]

    return X, y


def train(filepath: str):
    """Treina o modelo e retorna resultados."""

    print("Carregando e preparando dados...")
    X, y = load_and_prepare(filepath)

    print(f"Shape: {X.shape}")
    print(f"Distribuição da variável alvo:\n{y.value_counts()}")

    # Encode do target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Pipeline completo
    preprocessor = build_feature_pipeline()
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )

    print("\nTreinando modelo...")
    model.fit(X_train, y_train)

    # Avaliação
    from sklearn.metrics import classification_report, f1_score

    y_pred = model.predict(X_test)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nF1-score macro: {f1:.4f}")

    return model, le, f1


if __name__ == "__main__":
    model, le, f1 = train("data/raw/shelter_outcomes.csv")
