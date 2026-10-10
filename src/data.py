"""Etapa PREPARAR: carrega o dataset, divide em treino/validacao/teste e normaliza.

Cuidado importante (pedido no slide): a divisao acontece ANTES de ajustar o scaler.
O scaler aprende media e desvio somente com o treino, para nao haver vazamento
de informacao da validacao/teste.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLASSES = ["benigno", "maligno"]  # 0 = benigno, 1 = maligno


def prepare_data(cfg):
    """Retorna um dicionario com X/y de treino, validacao e teste + um resumo."""
    raw = load_breast_cancer()
    X = raw.data.astype(np.float32)
    # No sklearn 0 = maligno e 1 = benigno. Invertemos para 1 = maligno,
    # assim "recall" significa: dos tumores malignos, quantos o modelo achou.
    y = (1 - raw.target).astype(np.float32)

    # 1) separa o TESTE (fica guardado ate o final)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["seed"]
    )
    # 2) do que sobrou, separa a VALIDACAO
    val_frac = cfg["val_size"] / (1 - cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=cfg["seed"]
    )

    resumo = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic) - sklearn.datasets.load_breast_cancer",
        "n_features": int(X.shape[1]),
        "divisao": "estratificada, feita antes do scaler",
        "seed": cfg["seed"],
        "normalize": bool(cfg["normalize"]),
    }
    for nome, yy in (("treino", y_train), ("validacao", y_val), ("teste", y_test)):
        resumo[f"n_{nome}"] = int(len(yy))
        resumo[f"malignos_{nome}"] = int(yy.sum())

    if cfg["normalize"]:
        scaler = StandardScaler().fit(X_train)  # ajustado SOMENTE no treino
        X_train, X_val, X_test = (
            scaler.transform(a).astype(np.float32) for a in (X_train, X_val, X_test)
        )
        resumo["scaler"] = "StandardScaler ajustado apenas no treino"
        resumo["scaler_media"] = [round(float(v), 4) for v in scaler.mean_]
        resumo["scaler_desvio"] = [round(float(v), 4) for v in scaler.scale_]
    else:
        resumo["scaler"] = "nenhum (features na escala original)"

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "n_features": int(X.shape[1]),
        "resumo": resumo,
    }