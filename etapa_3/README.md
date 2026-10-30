# Etapa 3 - Pré-processamento

Nesta etapa preparamos os dados da EDA para começar a modelagem.

## O que foi feito

- Conferimos as colunas, tipos, valores faltantes e algumas inconsistências básicas.
- Retiramos 125 repetições, comparando as características e a nota sem considerar o Id.
- Mantivemos o corte em 7 para definir os vinhos de alta qualidade.
- Separamos as 11 características da variável alvo.
- Dividimos os 1.018 registros em 814 para treino e 204 para teste, mantendo proporções próximas das classes.
- Mantivemos os outliers e preparamos o RobustScaler para modelos sensíveis à escala.
- Salvamos os dados e as informações da preparação em `resultados/`.

## Como executar

Abra `02_pre_processamento.ipynb` e execute as células em ordem, usando a raiz do repositório ou esta pasta como diretório de trabalho. O código está dividido em partes, com os achados e decisões ao longo do notebook.

Para instalar as bibliotecas, execute da raiz do repositório:

```powershell
python -m pip install -r etapa_3/requirements.txt
```

O arquivo `preprocessamento.py` reúne a mesma preparação em uma função, caso seja útil executar por script:

```powershell
python etapa_3/preprocessamento.py
```

## Arquivos gerados

Os pares `X_train`/`y_train` e `X_test`/`y_test` estão na escala original e seguem a mesma ordem de linhas. A pasta também guarda as duplicatas, os registros removidos, os limites de outliers do treino, a distribuição das classes e os Ids da divisão. O `manifesto.json` registra as configurações, versões e a identificação da base utilizada.

A base original foi preservada. A base limpa inclui Id e nota para consulta; essas colunas não devem entrar no modelo.

## Relação com a EDA

Consideramos a análise de [pripaulino](https://github.com/pripaulino/wine-quality-classification-ml/tree/5e76d28485bfcf445dd9244c02043b0d8a46f7ac), no notebook `analyse_vinhos.ipynb`, consultado em 19/10/2026. Ela acrescenta correlações, comparações entre classes, investigação dos extremos e duplicatas. O CSV tem o mesmo conteúdo tabular da base local.

Seguimos a decisão de manter os outliers e o corte em 7. Usamos os nomes originais das colunas e uma nova divisão após retirar as repetições. Como toda a base já foi explorada na EDA, o teste reservado agora não é completamente independente dessas análises. Essa limitação deve acompanhar os resultados futuros.

## Para continuar na etapa 4

Usar os arquivos de treino e teste sem mudar a ordem dos pares X/y. Colocar a escala dentro da Pipeline de cada modelo e fazer a validação cruzada estratificada apenas no treino. Comparar pelo menos dois classificadores e avaliar o desbalanceamento com métricas além da acurácia.

Vamos começar com as 11 características. Pesos de classe, novas variáveis e seleção de características podem ser comparados no treino. O teste deve ser consultado somente depois das escolhas de modelos e ajustes. Nenhum classificador foi treinado nesta etapa.
