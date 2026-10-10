"""PIPELINE DE TREINO: configurar -> preparar -> treinar -> validar.

Cada arquivo .yaml vira UMA run no MLflow, com parametros, metricas por epoca,
artefatos (modelo + resumo do pre-processamento) e um trace com as etapas.

Uso:
    python -m src.train --config configs/exp1_hiperparametros/run_a_referencia.yaml
    python -m src.train --config configs/exp1_hiperparametros
    python -m src.train --config configs/exp1_hiperparametros/run_a_referencia.yaml --epochs 100

O conjunto de TESTE nao e usado aqui. Ele so e usado em src/test_final.py,
uma unica vez, depois de escolher a run pela validacao.
"""
import argparse
import copy
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn

from src.config import find_configs, load_config
from src.data import prepare_data
from src.engine import evaluate, make_loader, plot_confusion_matrix, train_one_epoch
from src.model import MLP

TRACKING_URI = "sqlite:///mlflow.db"  # arquivo onde o MLflow guarda o historico


def set_seed(seed):
    # Fixa as sementes do NumPy e do PyTorch (as unicas usadas no treino).
    # A semente do modulo `random` do Python NAO e fixada de proposito: o MLflow
    # usa esse gerador para sortear os IDs dos traces, e com semente fixa todas
    # as runs receberiam o mesmo ID de trace.
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_pipeline(cfg, config_path):
    """Executa uma run completa e devolve as metricas da melhor epoca na validacao."""
    set_seed(cfg["seed"])
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run(run_name=cfg["run_name"]):
        # ---------- CONFIGURAR: o que foi pedido para esta run ----------
        mlflow.set_tags({
            "pergunta": cfg["pergunta"],
            "dataset": "Breast Cancer Wisconsin (Diagnostic)",
            "config_file": str(config_path),
        })
        mlflow.log_params({
            "lr": cfg["lr"],
            "weight_decay_L2": cfg["weight_decay"],
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "hidden_layers": str(cfg["hidden_layers"]),
            "seed": cfg["seed"],
            "test_size": cfg["test_size"],
            "val_size": cfg["val_size"],
            "normalize": cfg["normalize"],
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
        })
        mlflow.log_dict(cfg, "config.json")

        # O trace registra as etapas (spans), com duracao e erro se houver
        with mlflow.start_span(name="pipeline_treino") as root:
            root.set_inputs({"config": str(config_path), "run_name": cfg["run_name"]})

            # ---------- PREPARAR: split + scaler ----------
            with mlflow.start_span(name="preparar") as span:
                span.set_inputs({k: cfg[k] for k in ("seed", "test_size", "val_size", "normalize")})
                data = prepare_data(cfg)
                train_loader = make_loader(data["X_train"], data["y_train"],
                                           cfg["batch_size"], True, cfg["seed"])
                val_loader = make_loader(data["X_val"], data["y_val"],
                                         cfg["batch_size"], False, cfg["seed"])
                resumo = data["resumo"]
                span.set_outputs({k: resumo[k] for k in
                                  ("n_treino", "n_validacao", "n_teste", "scaler")})
            mlflow.log_dict(resumo, "preprocessamento/resumo.json")

            model = MLP(data["n_features"], cfg["hidden_layers"])
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                         weight_decay=cfg["weight_decay"])
            mlflow.log_param("n_parametros", sum(p.numel() for p in model.parameters()))

            # ---------- TREINAR: metricas por epoca ----------
            with mlflow.start_span(name="treinar") as span:
                span.set_inputs({k: cfg[k] for k in ("lr", "weight_decay", "epochs", "batch_size")})
                best = {"loss": float("inf")}
                best_state, best_epoch = None, 0
                for epoch in range(1, cfg["epochs"] + 1):
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
                    val, _ = evaluate(model, val_loader, criterion)
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val["loss"],
                        "val_accuracy": val["accuracy"],
                        "val_f1": val["f1"],
                    }, step=epoch)
                    if val["loss"] < best["loss"]:  # guarda o melhor ponto da validacao
                        best, best_epoch = val, epoch
                        best_state = copy.deepcopy(model.state_dict())
                    if epoch == 1 or epoch % 10 == 0:
                        print(f"  epoca {epoch:3d} | train_loss {train_loss:.4f} | "
                              f"val_loss {val['loss']:.4f} | val_acc {val['accuracy']:.4f}")
                span.set_outputs({"best_epoch": best_epoch,
                                  "ultimo_train_loss": train_loss,
                                  "ultimo_val_loss": val["loss"]})

            # ---------- VALIDAR: resultado final + modelo ----------
            with mlflow.start_span(name="validar") as span:
                model.load_state_dict(best_state)
                final, cm = evaluate(model, val_loader, criterion)
                mlflow.log_metric("best_epoch", best_epoch)
                for k, v in final.items():
                    mlflow.log_metric(f"best_val_{k}", v)
                fig = plot_confusion_matrix(cm, f"Validacao - {cfg['run_name']}")
                mlflow.log_figure(fig, "validacao/matriz_confusao.png")
                plt.close(fig)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "modelo.pt"
                    torch.save(model.state_dict(), path)
                    mlflow.log_artifact(str(path), artifact_path="modelo")
                span.set_outputs({"best_epoch": best_epoch, **final})

            root.set_outputs({"best_val_loss": final["loss"],
                              "best_val_accuracy": final["accuracy"]})

    print(f"  VALIDACAO -> loss {final['loss']:.4f} | acc {final['accuracy']:.4f} | "
          f"f1 {final['f1']:.4f} | recall {final['recall']:.4f} (melhor epoca: {best_epoch})")
    return final


def main():
    parser = argparse.ArgumentParser(description="Pipeline de treino com MLflow")
    parser.add_argument("--config", required=True,
                        help="arquivo .yaml de uma run OU pasta com varias .yaml")
    # opcionais: sobrescrevem o que esta no .yaml
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    overrides = {k: getattr(args, k) for k in
                 ("lr", "weight_decay", "epochs", "batch_size", "seed")}

    mlflow.set_tracking_uri(TRACKING_URI)
    resultados = {}
    for path in find_configs(args.config):
        cfg = load_config(path, overrides)
        print(f"\n> Experimento: {cfg['experiment_name']} | Run: {cfg['run_name']}")
        resultados[cfg["run_name"]] = run_pipeline(cfg, path)

    print("\nResumo (validacao):")
    for nome, m in resultados.items():
        print(f"  {nome:16s} val_loss={m['loss']:.4f}  val_acc={m['accuracy']:.4f}  "
              f"val_f1={m['f1']:.4f}")
    print("\nAbra o painel com:  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")


if __name__ == "__main__":
    main()