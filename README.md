# A1 de IA — Treinamento observável com MLflow

Neste trabalho eu treinei uma rede neural (MLP em PyTorch) para classificar
tumores de mama do dataset **Breast Cancer Wisconsin (Diagnostic)** e usei o
**MLflow** para registrar cada treino: configuração, métricas, arquivos gerados
e etapas da execução.

- Disciplina: Inteligência Artificial 2026/2 — Sistemas de Informação — UNITINS
- Aluna: Yasmin Dos Santos Gothe
- Vídeo: LINK_DO_VIDEO

## O problema

O objetivo é dizer se um tumor é **benigno (0)** ou **maligno (1)** a partir de
30 medidas do exame. O dataset tem 569 exames (212 malignos e 357 benignos) e
já vem no scikit-learn, pela função `load_breast_cancer`.

## De onde eu parti

Usei como base os scripts da disciplina, que estão no repositório do professor
([sousamaf/AI-Lab](https://github.com/sousamaf/AI-Lab)):

- `simple_net_breast_cancer.py`: peguei daqui o dataset, a rede (30 → 16 → 8 → 1
  com ReLU), o otimizador Adam, as 50 épocas e a semente 42.
- `mlp_torch_avaliacao.py`: peguei daqui as métricas de avaliação (perda e
  acurácia de validação, precisão, recall, F1 e matriz de confusão).
- `mlp_torch_regularizacao_L2.py`: peguei daqui a regularização L2 com
  `weight_decay=1e-5`, que usei na run C.

O que eu mudei para virar um pipeline:

| No script da aula | No meu projeto |
|---|---|
| Tudo em um arquivo só | Etapas separadas: configurar, preparar, treinar e validar |
| Valores escritos direto no código | Um arquivo `.yaml` para cada run |
| Só treino e teste (80/20) | Treino, validação e teste (70/15/15), estratificado |
| Treina com todos os dados de uma vez | Treina em lotes de 32 |
| Só a acurácia de treino no terminal | Métricas de treino e validação por época no MLflow |
| Gráficos do matplotlib na tela | Curvas, artefatos e traces no MLflow |
| `Sigmoid` + `BCELoss` | `BCEWithLogitsLoss`, que faz a mesma conta em uma função |
| 1 = benigno | 1 = maligno, para o recall mostrar quantos tumores malignos foram encontrados |

## Como o projeto está organizado

```
configs/                          cada .yaml é uma run
  exp1_hiperparametros/           runs A, B e C
  exp2_incidente_normalizacao/    runs D e E
src/
  config.py      lê os arquivos .yaml
  data.py        carrega os dados, divide e normaliza
  model.py       a rede neural
  engine.py      treina uma época e calcula as métricas
  train.py       o pipeline: configurar -> preparar -> treinar -> validar
  test_final.py  escolhe a melhor run e avalia uma vez no teste
docs/            prints do MLflow
```

## Como rodar

Precisa de Python 3.10 ou mais novo (eu usei o 3.12). As versões das bibliotecas
estão fixadas no `requirements.txt`.

```bash
git clone https://github.com/Yasmingothe/a1-mlflow-breast-cancer.git
cd a1-mlflow-breast-cancer
pip install -r requirements.txt

# Experimento 1: um comando roda as três configurações
python -m src.train --config configs/exp1_hiperparametros

# Experimento 2
python -m src.train --config configs/exp2_incidente_normalizacao

# Escolhe a melhor run do experimento 1 e avalia uma vez no teste
python -m src.test_final --experiment a1_exp1_hiperparametros

# Abre o painel do MLflow em http://127.0.0.1:5001
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Também dá para mudar um parâmetro sem editar arquivo:

```bash
python -m src.train --config configs/exp1_hiperparametros/run_a_referencia.yaml --epochs 100 --lr 0.005
```

## O que fica registrado em cada run

| Tipo | O que é gravado |
|---|---|
| Parâmetros | taxa de aprendizado, L2, épocas, lote, arquitetura, semente, divisão e normalização |
| Métricas por época | `train_loss`, `val_loss`, `val_accuracy` e `val_f1` |
| Métricas finais | `best_epoch`, `best_val_*` e, só na run escolhida, `test_*` |
| Artefatos | o modelo (`modelo.pt`), o resumo do pré-processamento, a configuração e as matrizes de confusão |
| Trace | as etapas `preparar`, `treinar` e `validar`, com o tempo de cada uma |

## Cuidados que eu tomei

- Dividi os dados **antes** de normalizar, e a divisão é estratificada (397 treino, 86 validação, 86 teste).
- O scaler aprende **só com o treino**, para não vazar informação da validação e do teste.
- A semente é fixa (42), então rodar de novo dá o mesmo resultado.
- Escolhi o modelo olhando **só a validação**. O teste foi usado **uma única vez**, no modelo escolhido.

## Experimento 1 — hiperparâmetros

**Pergunta:** qual combinação de taxa de aprendizado e regularização L2 dá o
melhor resultado na validação?

As três runs usam a mesma rede, a mesma divisão, a mesma semente e 50 épocas.
Só mudam a taxa e o L2.

| Run | Taxa | L2 | best_val_loss | Acurácia (val) | F1 (val) | Melhor época |
|---|---|---|---|---|---|---|
| A — referência | 0.01 | 0 | 0.0642 | 0.9884 | 0.9841 | 15 |
| B — taxa menor | 0.001 | 0 | 0.0685 | 0.9767 | 0.9688 | 36 |
| C — com L2 | 0.01 | 1e-5 | **0.0529** | 0.9884 | 0.9841 | 17 |

**Escolhi a run C**, porque teve a menor perda de validação.

O que eu observei:

- A e C acertam 85 de 86 exames na validação, e a B acerta 84. Com tão poucos
  exames, um erro a mais muda a acurácia em cerca de 1,2 ponto. Por isso a
  diferença entre as três **não é conclusiva**, e usei a perda de validação
  para desempatar.
- As runs A e C têm **overfitting**: depois da época 15–17 a perda de treino
  chega perto de zero, mas a perda de validação volta a subir. O L2 de 1e-5 foi
  pequeno demais para evitar isso. Por esse motivo o pipeline guarda o modelo
  da melhor época de validação, e não o da última.
- A run B aprende mais devagar, mas fica mais estável: em 50 épocas a perda de
  validação ainda não tinha subido.

### Teste final (run C, uma única vez)

| Acurácia | Precisão | Recall | F1 |
|---|---|---|---|
| 0.9884 | 1.0000 | 0.9688 | 0.9841 |

| | Previsto benigno | Previsto maligno |
|---|---|---|
| **Real benigno** | 54 | 0 |
| **Real maligno** | 1 | 31 |

O modelo errou 1 exame em 86. Foi um falso negativo: um tumor maligno
classificado como benigno, que é o erro mais grave nesse tipo de problema.

## Experimento 2 — diagnóstico do incidente

**Pergunta:** por que o treino fica ruim sem normalizar os dados, e a
normalização resolve?

| Run | Normalização | Perda na época 1 | Acurácia (val) | Recall (val) |
|---|---|---|---|---|
| D — sem scaler | não | 4.05 | 0.9302 | 0.8438 |
| E — com scaler | sim | 0.56 | 0.9884 | 0.9688 |

- **Sintoma:** sem normalizar, o treino começa com a perda muito alta (4,05) e
  acurácia de 0,37, e a curva de validação fica oscilando.
- **Hipótese:** as medidas têm escalas muito diferentes. A área passa de 2.000 e
  a suavidade fica perto de 0,1, então as maiores dominam o ajuste dos pesos.
- **Nova run:** com o StandardScaler, a curva fica suave e a acurácia sobe de
  0,93 para 0,99. O recall sobe de 0,84 para 0,97, ou seja, o modelo deixa de
  perder cerca de 16% dos tumores malignos e passa a perder cerca de 3%.

Os prints do MLflow estão na pasta `docs/`.

## Uso de IA generativa

Usei IA generativa (Claude, da Anthropic) como apoio para portar os scripts da
disciplina para o formato de pipeline e para instrumentar o código com MLflow.
Eu rodei os experimentos, conferi os resultados no MLflow e fiz a análise.