"""Etapa 4 - Gradient Boosting.

Treina e avalia o GradientBoostingClassifier usando o mesmo split da etapa 3
(etapa_3/resultados). Pode ser executado pela raiz do repositório:

    python src/gradient_boosting.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "etapa_3" / "resultados"
SAIDA = RAIZ / "results" / "gradient_boosting"

# Grade pequena para rodar rápido: 16 combinações x 5 folds
GRADE = {
    "modelo__n_estimators": [100, 200],
    "modelo__learning_rate": [0.05, 0.1],
    "modelo__max_depth": [2, 3],
    "modelo__subsample": [0.8, 1.0],
}


class GradientBoostingBalanceado(GradientBoostingClassifier):
    """GradientBoostingClassifier que calcula os pesos das classes dentro do fit.

    O GradientBoosting não tem class_weight. Calculando o sample_weight aqui,
    os pesos usam apenas os dados de treino de cada ajuste (inclusive em cada
    fold da validação cruzada), sem olhar para a validação ou para o teste.
    """

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight)


def carregar_dados(pasta=DADOS):
    """Lê os conjuntos de treino e teste salvos na etapa 3."""
    X_train = pd.read_csv(pasta / "X_train.csv")
    X_test = pd.read_csv(pasta / "X_test.csv")
    y_train = pd.read_csv(pasta / "y_train.csv")["quality_class"]
    y_test = pd.read_csv(pasta / "y_test.csv")["quality_class"]
    return X_train, X_test, y_train, y_test


def criar_pipeline(**parametros):
    """Mesma pipeline da etapa 3 (RobustScaler) com o classificador no final.

    Para árvores a escala não muda o resultado, mas mantemos a mesma estrutura
    dos outros modelos para a comparação ficar justa.
    """
    return Pipeline([
        ("escala", RobustScaler()),
        ("modelo", GradientBoostingBalanceado(random_state=SEED, **parametros)),
    ])


def validacao_cruzada():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def treinar_baseline(X_train, y_train):
    """Baseline com os parâmetros padrão e F1 médio na validação cruzada."""
    modelo = criar_pipeline()
    cv = cross_validate(modelo, X_train, y_train, cv=validacao_cruzada(), scoring="f1")
    modelo.fit(X_train, y_train)
    return modelo, cv["test_score"].mean()


def ajustar_hiperparametros(X_train, y_train, grade=GRADE):
    """Busca em grade usando apenas o treino; escolhe pelo F1 da classe 1."""
    busca = GridSearchCV(
        criar_pipeline(), grade, scoring="f1", cv=validacao_cruzada(), n_jobs=-1
    )
    busca.fit(X_train, y_train)
    return busca


def avaliar(modelo, X_test, y_test):
    """Métricas no conjunto de teste (classe positiva = alta qualidade)."""
    previsto = modelo.predict(X_test)
    probabilidade = modelo.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, previsto).ravel()
    return {
        "accuracy": accuracy_score(y_test, previsto),
        "precision_1": precision_score(y_test, previsto),
        "recall_1": recall_score(y_test, previsto),
        "f1_1": f1_score(y_test, previsto),
        "roc_auc": roc_auc_score(y_test, probabilidade),
        "average_precision": average_precision_score(y_test, probabilidade),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def importancia_variaveis(modelo, colunas):
    """feature_importances_ do classificador, da maior para a menor."""
    importancias = pd.Series(
        modelo.named_steps["modelo"].feature_importances_,
        index=pd.Index(colunas, name="variavel"), name="importancia"
    )
    return importancias.sort_values(ascending=False)


def salvar_graficos(modelo, X_test, y_test, importancias, pasta=SAIDA):
    """Matriz de confusão, curva ROC e importância das variáveis em PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        modelo, X_test, y_test, display_labels=["Baixa/média", "Alta"], cmap="Blues", ax=ax
    )
    ax.set_title("Gradient Boosting - matriz de confusão (teste)")
    fig.tight_layout()
    fig.savefig(pasta / "matriz_confusao.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(modelo, X_test, y_test, name="Gradient Boosting", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Gradient Boosting - curva ROC (teste)")
    fig.tight_layout()
    fig.savefig(pasta / "curva_roc.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    importancias.sort_values().plot.barh(ax=ax, color="tab:blue")
    ax.set_xlabel("Importância")
    ax.set_title("Gradient Boosting - importância das variáveis")
    fig.tight_layout()
    fig.savefig(pasta / "importancia_variaveis.png", dpi=120)
    plt.close(fig)


def executar(pasta=SAIDA):
    """Roda todo o fluxo e salva os resultados em results/gradient_boosting."""
    pasta.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = carregar_dados()

    baseline, f1_cv_baseline = treinar_baseline(X_train, y_train)
    busca = ajustar_hiperparametros(X_train, y_train)
    melhor = busca.best_estimator_

    metricas = pd.DataFrame({
        "baseline": {"f1_cv_treino": f1_cv_baseline, **avaliar(baseline, X_test, y_test)},
        "ajustado": {"f1_cv_treino": busca.best_score_, **avaliar(melhor, X_test, y_test)},
    }).T
    metricas.index.name = "modelo"
    metricas.to_csv(pasta / "metricas.csv")

    melhores = {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()}
    (pasta / "melhores_hiperparametros.json").write_text(
        json.dumps(melhores, indent=2), encoding="utf-8"
    )

    importancias = importancia_variaveis(melhor, X_train.columns)
    importancias.to_csv(pasta / "importancia_variaveis.csv")
    salvar_graficos(melhor, X_test, y_test, importancias, pasta)

    return metricas, melhores, importancias


if __name__ == "__main__":
    matplotlib.use("Agg")
    metricas, melhores, _ = executar()
    print("Melhores hiperparâmetros:", melhores)
    print(metricas.round(3).to_string())
