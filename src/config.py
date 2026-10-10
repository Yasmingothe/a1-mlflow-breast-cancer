"""Etapa CONFIGURAR: le os arquivos .yaml (um arquivo = uma run)."""
from pathlib import Path

import yaml

OBRIGATORIOS = [
    "experiment_name", "run_name", "seed", "test_size", "val_size", "normalize",
    "hidden_layers", "lr", "weight_decay", "epochs", "batch_size",
]


def find_configs(path):
    """Aceita um arquivo .yaml ou uma pasta (roda todos os .yaml dela, em ordem)."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.yaml"))
        if not files:
            raise FileNotFoundError(f"Nenhum .yaml encontrado em {p}")
        return files
    if not p.exists():
        raise FileNotFoundError(f"Config nao encontrada: {p}")
    return [p]


def load_config(path, overrides=None):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    faltando = [k for k in OBRIGATORIOS if k not in cfg]
    if faltando:
        raise ValueError(f"{path}: faltam as chaves {faltando}")
    cfg.setdefault("pergunta", "")
    # valores passados na linha de comando sobrescrevem o .yaml
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    cfg["lr"] = float(cfg["lr"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    return cfg