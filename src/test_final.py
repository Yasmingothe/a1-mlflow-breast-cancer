"""TESTE FINAL: escolhe a melhor run pela VALIDACAO e avalia UMA vez no teste.

Uso:
    python -m src.test_final --experiment a1_exp1_hiperparametros

O criterio de escolha e o menor best_val_loss. As metricas de teste e a matriz
de confusao sao gravadas dentro da propria run escolhida (tag escolhida=sim).
"""
import argparse

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn

from src.data import prepare_data
from src.engine import evaluate, make_loader, plot_confusion_matrix
from src.model import MLP
from src.train import TRACKING_URI

CRITERIO = "metrics.best_val_loss"


def main():
    parser = argparse.ArgumentParser(description="Avaliacao final no conjunto de teste")
    parser.add_argument("--experiment", required=True, help="nome do experimento no MLflow")
    parser.add_argument("--force", action="store_true",
                        help="refaz o teste mesmo se ja tiver sido feito")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[args.experiment])
    if runs.empty or CRITERIO not in runs.columns:
        raise SystemExit(f"Nenhuma run treinada no experimento '{args.experiment}'. "
                         "Rode o src.train primeiro.")
    runs = runs.dropna(subset=[CRITERIO]).sort_values(CRITERIO)

    print(f"\nRuns do experimento {args.experiment} (ordenadas por val_loss):")
    for _, r in runs.iterrows():
        print(f"  {r['tags.mlflow.runName']:16s} best_val_loss={r[CRITERIO]:.4f}  "
              f"best_val_accuracy={r['metrics.best_val_accuracy']:.4f}")

    if "metrics.test_accuracy" in runs.columns and runs["metrics.test_accuracy"].notna().any():
        if not args.force:
            raise SystemExit("\nO teste ja foi feito neste experimento (ele deve ser usado "
                             "uma unica vez). Use --force se realmente quiser refazer.")

    escolhida = runs.iloc[0]
    run_id, nome = escolhida["run_id"], escolhida["tags.mlflow.runName"]
    print(f"\nRun escolhida pela validacao: {nome} ({run_id})")

    # reconstroi exatamente os mesmos dados e o mesmo modelo da run escolhida
    cfg = mlflow.artifacts.load_dict(f"runs:/{run_id}/config.json")
    data = prepare_data(cfg)
    test_loader = make_loader(data["X_test"], data["y_test"], cfg["batch_size"], False, cfg["seed"])
    model = MLP(data["n_features"], cfg["hidden_layers"])
    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="modelo/modelo.pt")
    model.load_state_dict(torch.load(local))

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_span(name="teste_final") as span:
            span.set_inputs({"run_escolhida": nome, "criterio": "menor best_val_loss"})
            test, cm = evaluate(model, test_loader, nn.BCEWithLogitsLoss())
            span.set_outputs(test)
        for k, v in test.items():
            mlflow.log_metric(f"test_{k}", v)
        fig = plot_confusion_matrix(cm, f"Teste - {nome}")
        mlflow.log_figure(fig, "teste/matriz_confusao.png")
        plt.close(fig)
        mlflow.set_tag("escolhida", "sim")

    print(f"TESTE -> acc {test['accuracy']:.4f} | precision {test['precision']:.4f} | "
          f"recall {test['recall']:.4f} | f1 {test['f1']:.4f}")
    print(f"Matriz de confusao (linhas = real, colunas = previsto):\n{cm}")


if __name__ == "__main__":
    main()