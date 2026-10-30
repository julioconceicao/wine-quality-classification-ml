"""Rotina auxiliar com os mesmos passos do notebook de pré-processamento."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

PASTA = Path(__file__).resolve().parent
FONTE = PASTA.parent / 'data' / 'WineQT.csv'
FEATURES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
            'density', 'pH', 'sulphates', 'alcohol']
SEED = 42


def criar_preprocessador():
    """Cria uma nova pipeline para usar com cada modelo, ainda sem ajuste."""
    return Pipeline([('escala', RobustScaler())])


def preparar(caminho=FONTE, saida=None):
    """Confere a base, retira repetições e salva os conjuntos de treino e teste."""
    caminho = Path(caminho)
    saida = Path(saida) if saida else PASTA / 'resultados'
    saida.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(caminho)
    obrigatorias = FEATURES + ['quality', 'Id']
    faltando = set(obrigatorias) - set(original.columns)
    if faltando:
        raise ValueError(f'Colunas ausentes: {sorted(faltando)}')
    df = original[obrigatorias].copy()
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in obrigatorias):
        raise ValueError('As colunas esperadas devem ser numéricas.')
    nulos = df.isna().sum().rename('nulos')
    nulos.to_csv(saida / 'auditoria_nulos.csv', encoding='utf-8')
    if nulos.sum() or not np.isfinite(df.to_numpy()).all():
        raise ValueError('Dados ausentes ou não finitos: investigar antes de continuar.')
    if not df['quality'].between(0, 10).all() or not (df['quality'] % 1 == 0).all():
        raise ValueError('Notas devem ser inteiras entre 0 e 10.')
    if not df['Id'].is_unique:
        raise ValueError('Identificadores repetidos: investigar a fonte.')
    if (df[FEATURES] < 0).any().any() or (df[['density', 'pH']] <= 0).any().any():
        raise ValueError('Valores incompatíveis com as verificações básicas de domínio.')
    if (df['free sulfur dioxide'] > df['total sulfur dioxide']).any():
        raise ValueError('SO2 livre maior que o total: investigar registros.')

    comparacao = FEATURES + ['quality']
    repetidas = df.duplicated(comparacao, keep='first')
    envolvidas = df.duplicated(comparacao, keep=False)
    df.loc[envolvidas].sort_values(comparacao).to_csv(saida / 'grupos_duplicados.csv', index=False)
    df.loc[repetidas].to_csv(saida / 'registros_removidos.csv', index=False)
    limpa = df.loc[~repetidas].copy()
    # Mesmo perfil com notas distintas deve ser investigado, não dividido aleatoriamente.
    if limpa.duplicated(FEATURES).any():
        raise ValueError('Perfis idênticos com notas distintas: usar divisão por grupos ou investigar.')
    limpa['quality_class'] = (limpa['quality'] >= 7).astype('int64')
    treino_idx, teste_idx = train_test_split(
        limpa.index, test_size=0.20, stratify=limpa['quality_class'], random_state=SEED)
    treino, teste = limpa.loc[treino_idx], limpa.loc[teste_idx]
    X_train, X_test = treino[FEATURES].copy(), teste[FEATURES].copy()
    y_train, y_test = treino['quality_class'].copy(), teste['quality_class'].copy()
    assert set(treino.Id).isdisjoint(teste.Id)
    assert set(map(tuple, X_train.to_numpy())).isdisjoint(map(tuple, X_test.to_numpy()))

    # Usamos só o treino para calcular os limites e mantemos os valores extremos.
    q1, q3 = X_train.quantile(.25), X_train.quantile(.75)
    iqr = q3 - q1
    baixo, alto = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = pd.DataFrame({'q1_treino': q1, 'q3_treino': q3,
                            'limite_inferior': baixo, 'limite_superior': alto,
                            'outliers_treino': ((X_train < baixo) | (X_train > alto)).sum()})
    outliers.index.name = 'variavel'
    outliers.to_csv(saida / 'outliers_treino.csv')
    distribuicao = pd.concat({nome: y.value_counts().reindex([0, 1], fill_value=0)
                             for nome, y in [('base_limpa', limpa.quality_class),
                                             ('treino', y_train), ('teste', y_test)]}, axis=1)
    distribuicao.index.name = 'quality_class'
    distribuicao.to_csv(saida / 'distribuicao_classes.csv')
    for nome, dados in [('X_train', X_train), ('X_test', X_test),
                        ('y_train', y_train), ('y_test', y_test)]:
        dados.to_csv(saida / f'{nome}.csv', index=False)
    pd.concat([treino[['Id']].assign(conjunto='treino'),
               teste[['Id']].assign(conjunto='teste')]).to_csv(saida / 'divisao_por_id.csv', index=False)
    limpa.to_csv(saida / 'base_sem_duplicatas.csv', index=False)
    manifesto = dict(fonte_relativa='../data/WineQT.csv',
        sha256=hashlib.sha256(caminho.read_bytes()).hexdigest(),
        seed=SEED, proporcao_teste=.20, alvo='quality >= 7', features=FEATURES,
        registros_originais=len(df), repeticoes_removidas=int(repetidas.sum()),
        registros_em_grupos_duplicados=int(envolvidas.sum()), registros_limpos=len(limpa),
        registros_treino=len(treino), registros_teste=len(teste),
        outliers='mantidos; limites IQR calculados no treino',
        escala='RobustScaler não ajustado; encaixar na pipeline do classificador',
        versoes={'pandas': pd.__version__, 'numpy': np.__version__, 'scikit-learn': sklearn.__version__})
    (saida / 'manifesto.json').write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding='utf-8')
    return dict(original=original, limpa=limpa, repetidas=repetidas, envolvidas=envolvidas,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                distribuicao=distribuicao, outliers=outliers, manifesto=manifesto)


if __name__ == '__main__':
    print(json.dumps(preparar()['manifesto'], ensure_ascii=False, indent=2))
