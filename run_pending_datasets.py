#!/usr/bin/env python3
"""
run_pending_datasets.py
=========================
Orquestrador "burro-em-cima, esperto-embaixo": pra cada arquivo em run/ que
representa um dataset COM instance selection, faz exatamente 2 coisas, na
ordem certa, e so isso:

  1. GERA o dataset reduzido (chamando o run_instance_selection.py correto
     -- instanceselection-main pra LSSm/CNN, e2sc-is-main pro E2SC,
     bio-is-main pro bio-is), se ainda nao existir.
  2. RODA `bash run/<dataset>_is_<metodo>.sh 0 9`, se ainda nao tiver
     resultado final.

RODAR NA RAIZ DO REPOSITORIO RAG-Fuse.

Como ele decide quais combinacoes processar
--------------------------------------------
Ele NAO tem uma lista hardcoded de datasets. Ele escaneia `run/*.sh`,
reconhece os sufixos de metodo conhecidos (_is_lssm, _is_cnn,
_is_e2sc_r20/r25/r30/iterative, _is_bio_b25_t50) e extrai o dataset base
de cada nome de arquivo. Datasets cujo nome comeca com algum prefixo em
--exclude (default: acm, ohsumed, reut90) sao pulados -- pra voce que ja
processou esses tres manualmente.

Isso significa: se amanha voce adicionar `run/livedoor_is_cnn.sh`, o script
ja pega ele automaticamente na proxima execucao, sem precisar editar nada
aqui.

Idempotencia (pode rodar de novo sem medo)
--------------------------------------------
Antes de gerar, checa se `resource/dataset/<dataset>_is_<metodo>/fold_9/
train.pkl` ja existe -- se sim, pula a geracao.
Antes de rodar o pipeline, checa se `resource/result/Aggregated_<model.name>
_<dataset>_is_<metodo>/..._9.rts` ja existe -- se sim, pula o pipeline.
Ou seja: da pra interromper (Ctrl+C) e rodar de novo depois que ele retoma
de onde parou, sem reprocessar o que ja esta pronto.

Falha em uma combinacao NAO para o lote inteiro
--------------------------------------------------
Se a geracao ou o pipeline de um par (dataset, metodo) falhar, o script
registra o erro, segue pro proximo par, e no final imprime um resumo com
tudo que deu certo/errado -- do mesmo jeito que os outros scripts desta
conversa.

Exemplos
--------
# So mostrar o plano, sem executar nada:
python run_pending_datasets.py --dry-run

# Rodar de verdade:
python run_pending_datasets.py \\
    --lssm-cnn-script /caminho/instanceselection-main/run_instance_selection.py \\
    --lssm-cnn-iselib-dir /caminho/instanceselection-main \\
    --e2sc-script /caminho/e2sc-is-main/run_instance_selection.py \\
    --e2sc-iselib-dir /caminho/e2sc-is \\
    --bio-script /caminho/bio-is-main/run_instance_selection.py \\
    --bio-iselib-dir /caminho/bio-is

# So um subconjunto (teste antes do lote completo):
python run_pending_datasets.py --only-datasets dblp --only-methods lssm cnn ...
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Mapeamento sufixo-de-arquivo -> como gerar aquele dataset
# ---------------------------------------------------------------------------
# generator: qual dos 3 scripts de instance selection usar
# extra_args: flags especificas do metodo, passadas ao gerador
IS_METHODS = {
    "lssm":            {"generator": "lssm_cnn", "extra_args": ["-m", "lssm"]},
    "cnn":             {"generator": "lssm_cnn", "extra_args": ["-m", "cnn"]},
    "e2sc_r20":        {"generator": "e2sc", "extra_args": ["--reduction-rate", "0.20"]},
    "e2sc_r25":        {"generator": "e2sc", "extra_args": ["--reduction-rate", "0.25"]},
    "e2sc_r30":        {"generator": "e2sc", "extra_args": ["--reduction-rate", "0.30"]},
    "e2sc_iterative":  {"generator": "e2sc", "extra_args": ["--beta-mode", "iterative"]},
    "bio_b25_t50":     {"generator": "bio", "extra_args": ["--redundancy-rate", "0.25", "--noise-rate", "0.50"]},
}
# ordenado do sufixo mais longo pro mais curto, pra bater primeiro o mais especifico
METHOD_SUFFIXES = sorted(IS_METHODS.keys(), key=len, reverse=True)


def discover_pending(run_dir: Path, exclude_prefixes: list):
    """Escaneia run/*.sh e retorna lista de (dataset, method) a processar."""
    pending = []
    for sh in sorted(run_dir.glob("*.sh")):
        stem = sh.stem  # ex: "dblp_is_e2sc_r20"
        if "_is_" not in stem:
            continue  # scripts sem IS (ex: dblp.sh) nao entram, so as variantes
        for method in METHOD_SUFFIXES:
            suffix = f"_is_{method}"
            if stem.endswith(suffix):
                dataset = stem[: -len(suffix)]
                if any(dataset == p or dataset.startswith(p + "_") for p in exclude_prefixes) or dataset in exclude_prefixes:
                    break
                pending.append((dataset, method))
                break
    return pending


# ---------------------------------------------------------------------------
# Parse de model.name do proprio run/<dataset>_is_<metodo>.sh (pra saber o
# nome esperado da pasta de resultado final, e checar se ja foi feito)
# ---------------------------------------------------------------------------
def parse_model_name(run_script_path: Path) -> str:
    text = run_script_path.read_text()
    m = re.search(r"^model=(\S+)", text, re.MULTILINE)
    if not m:
        return None
    model_base = m.group(1)
    names = set(re.findall(r"model\.name=(\S+?)\s*(?:\\|$)", text, re.MULTILINE))
    names = {n.replace("${model}", model_base) for n in names}
    return sorted(names)[0] if len(names) == 1 else (sorted(names)[0] if names else None)


def dataset_already_generated(ragfuse_dir: Path, dataset: str, method: str) -> bool:
    marker = ragfuse_dir / "resource" / "dataset" / f"{dataset}_is_{method}" / "fold_9" / "train.pkl"
    return marker.exists() and marker.stat().st_size > 0


def pipeline_already_done(ragfuse_dir: Path, dataset: str, method: str, model_name: str) -> bool:
    if not model_name:
        return False
    full_name = f"{dataset}_is_{method}"
    marker = (ragfuse_dir / "resource" / "result" / f"Aggregated_{model_name}_{full_name}"
              / f"Aggregated_{model_name}_{full_name}_9.rts")
    return marker.exists() and marker.stat().st_size > 0


# ---------------------------------------------------------------------------
# Execucao dos dois passos
# ---------------------------------------------------------------------------
def build_generate_cmd(method: str, dataset: str, args: argparse.Namespace):
    cfg = IS_METHODS[method]
    gen = cfg["generator"]
    if gen == "lssm_cnn":
        script, iselib_dir = args.lssm_cnn_script, args.lssm_cnn_iselib_dir
    elif gen == "e2sc":
        script, iselib_dir = args.e2sc_script, args.e2sc_iselib_dir
    elif gen == "bio":
        script, iselib_dir = args.bio_script, args.bio_iselib_dir
    else:
        raise ValueError(gen)

    if script is None or iselib_dir is None:
        return None  # caminhos nao configurados pra esse gerador

    cmd = [
        sys.executable, str(script),
        "--ragfuse-dir", str(args.ragfuse_dir),
        "--iselib-dir", str(iselib_dir),
        "--datasets", dataset,
        "--update-config",
    ] + cfg["extra_args"]
    return cmd


def run_cmd(cmd, log_path: Path, cwd: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        logf.write(f"$ {' '.join(cmd)}\n\n")
        logf.flush()
        result = subprocess.run(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT)
    return result.returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ragfuse-dir", type=Path, default=Path.cwd(),
                    help="Raiz do RAG-Fuse (default: diretorio atual).")

    p.add_argument("--lssm-cnn-script", type=Path, default=None,
                    help="Caminho pro run_instance_selection.py do repo instanceselection-main (LSSm/CNN).")
    p.add_argument("--lssm-cnn-iselib-dir", type=Path, default=None,
                    help="Raiz do repo instanceselection-main.")

    p.add_argument("--e2sc-script", type=Path, default=None,
                    help="Caminho pro run_instance_selection.py adaptado pro E2SC.")
    p.add_argument("--e2sc-iselib-dir", type=Path, default=None,
                    help="Raiz do repo e2sc-is-main.")

    p.add_argument("--bio-script", type=Path, default=None,
                    help="Caminho pro run_instance_selection.py adaptado pro bio-is.")
    p.add_argument("--bio-iselib-dir", type=Path, default=None,
                    help="Raiz do repo bio-is-main.")

    p.add_argument("--exclude", nargs="+", default=["acm", "ohsumed", "reut90"],
                    help="Prefixos de dataset a NAO processar (default: acm ohsumed reut90).")
    p.add_argument("--only-datasets", nargs="+", default=None,
                    help="Se informado, processa so esses datasets (apos aplicar --exclude).")
    p.add_argument("--only-methods", nargs="+", default=None, choices=list(IS_METHODS.keys()),
                    help="Se informado, processa so esses metodos.")

    p.add_argument("--start-fold", type=int, default=0)
    p.add_argument("--end-fold", type=int, default=9)
    p.add_argument("--log-dir", type=Path, default=None,
                    help="Onde salvar os logs de cada etapa. Default: <ragfuse-dir>/.pending_run_logs/")
    p.add_argument("--dry-run", action="store_true",
                    help="So mostra o plano (o que seria gerado/rodado), sem executar nada.")
    return p.parse_args()


def main():
    args = parse_args()
    ragfuse_dir = args.ragfuse_dir.resolve()
    run_dir = ragfuse_dir / "run"
    log_dir = (args.log_dir or (ragfuse_dir / ".pending_run_logs")).resolve()

    if not run_dir.is_dir():
        raise SystemExit(f"NAO achei {run_dir}. Rode com --ragfuse-dir apontando pra raiz do RAG-Fuse.")

    pending = discover_pending(run_dir, args.exclude)
    if args.only_datasets:
        pending = [(d, m) for d, m in pending if d in args.only_datasets]
    if args.only_methods:
        pending = [(d, m) for d, m in pending if m in args.only_methods]

    if not pending:
        print("Nada pra fazer: nenhuma combinacao dataset+metodo encontrada em run/ "
              "depois de aplicar --exclude/--only-datasets/--only-methods.")
        return

    print(f"Encontradas {len(pending)} combinacoes dataset+metodo (excluindo {args.exclude}):")
    for d, m in pending:
        print(f"  - {d}_is_{m}")
    print()

    if args.dry_run:
        print("--dry-run: nada foi executado. Confira os caminhos dos geradores (--lssm-cnn-script etc.) "
              "e remova --dry-run pra rodar de verdade.")
        return

    results = []
    for dataset, method in pending:
        full_name = f"{dataset}_is_{method}"
        run_script = run_dir / f"{full_name}.sh"
        if not run_script.exists():
            print(f"[{full_name}] AVISO: {run_script} nao existe, pulando (inconsistente com o discover).")
            continue

        model_name = parse_model_name(run_script)
        row = {"dataset": dataset, "method": method, "gerado": None, "pipeline": None}

        # --- Passo 1: gerar o dataset com IS ---------------------------------
        if dataset_already_generated(ragfuse_dir, dataset, method):
            print(f"[{full_name}] dataset ja existe, pulando geracao.")
            row["gerado"] = "ja existia"
        else:
            cmd = build_generate_cmd(method, dataset, args)
            if cmd is None:
                print(f"[{full_name}] ERRO: caminho do gerador nao configurado pro metodo '{method}' "
                      f"(confira --lssm-cnn-script/--e2sc-script/--bio-script). Pulando.")
                row["gerado"] = "SEM GERADOR CONFIGURADO"
                results.append(row)
                continue

            log_path = log_dir / f"{full_name}__generate.log"
            print(f"[{full_name}] gerando dataset... (log: {log_path})")
            rc = run_cmd(cmd, log_path, cwd=ragfuse_dir)
            if rc != 0:
                print(f"[{full_name}] ERRO na geracao (codigo {rc}). Veja {log_path}. Pulando pipeline.")
                row["gerado"] = f"FALHOU (rc={rc})"
                results.append(row)
                continue
            row["gerado"] = "OK"

        # --- Passo 2: rodar o pipeline ----------------------------------------
        if pipeline_already_done(ragfuse_dir, dataset, method, model_name):
            print(f"[{full_name}] pipeline ja tem resultado final, pulando.")
            row["pipeline"] = "ja existia"
        else:
            log_path = log_dir / f"{full_name}__pipeline.log"
            cmd = ["bash", str(run_script), str(args.start_fold), str(args.end_fold)]
            print(f"[{full_name}] rodando pipeline (folds {args.start_fold}-{args.end_fold})... (log: {log_path})")
            rc = run_cmd(cmd, log_path, cwd=ragfuse_dir)
            row["pipeline"] = "OK" if rc == 0 else f"FALHOU (rc={rc})"
            if rc != 0:
                print(f"[{full_name}] ERRO no pipeline (codigo {rc}). Veja {log_path}.")

        results.append(row)
        print()

    print("\n=== Resumo final ===")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
