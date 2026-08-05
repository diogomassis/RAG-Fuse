#!/usr/bin/env python3
"""
compute_results.py
-------------------
Roda (opcionalmente) o pipeline do RAG-Fuse para um dataset e agrega os
resultados de todos os folds na tabela final:

    nDCG x100 (Tail label / Head label @1,@5,@10)
    Precision x100 (Tail label / Head label @1,@5,@10)
    Mac-F1 x100 (@1,@5,@10)
    Mic-F1 x100 (@1,@5,@10)

EXECUTAR NA RAIZ DO REPOSIToRIO RAG-Fuse (mesmo lugar onde ficam `run/`,
`resource/`, `main.py`, `run.sh`).

--------------------------------------------------------------------------
COMO USAR
--------------------------------------------------------------------------

1) So agregar resultados que ja existem em resource/ (nao roda nada):
   python compute_results.py --dataset reut90_is_lssm

2) Rodar o pipeline inteiro (folds 0 a 9) e depois agregar:
   python compute_results.py --dataset reut90_is_lssm --run --start-fold 0 --end-fold 9

3) So checar se todas as etapas/folds geraram os arquivos esperados,
   sem agregar nada (diagnostico rapido de falhas):
   python compute_results.py --dataset reut90_is_lssm --check-only

O script NUNCA confia nos arquivos resource/time/*.tmr (eles sao escritos
mesmo quando uma etapa falha, ja que os scripts em run/*.sh nao usam
`set -e`). Em vez disso, ele confere a existencia/tamanho dos artefatos
reais de cada etapa (checkpoint, .prd, .rts, .rnk).

--------------------------------------------------------------------------
SAIDA
--------------------------------------------------------------------------
- Tabela impressa no terminal (mesmo layout da planilha: nDCG/Precision
  tail+head @1/5/10, Mac-F1/Mic-F1 @1/5/10), no formato "media(desvio)".
- CSV salvo em resource/result/<model_name>_<dataset>_final_table.csv
"""

import argparse
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

THRESHOLDS = [1, 5, 10]
STAGES = ["sparse_retrieve", "fit", "predict", "eval", "fuse", "aggregate"]


def parse_run_script(dataset: str, repo_root: Path):
    """Extrai model base (ex: RetrieverBERT) e o model.name usado no
    pipeline (ex: LLM_RetrieverBERT), lendo diretamente o script run/<dataset>.sh
    em vez de assumir um valor fixo."""
    script_path = repo_root / "run" / f"{dataset}.sh"
    if not script_path.exists():
        sys.exit(f"[ERRO] Nao encontrei {script_path}. Dataset invalido?")

    text = script_path.read_text()

    m = re.search(r"^model=(\S+)", text, re.MULTILINE)
    if not m:
        sys.exit(f"[ERRO] Nao consegui achar a linha 'model=' em {script_path}")
    model_base = m.group(1)

    # Todas as ocorrencias de model.name= usadas no script (fit/predict/eval/fuse/aggregate)
    model_names = set(re.findall(r"model\.name=(\S+?)\s*(?:\\|$)", text, re.MULTILINE))
    model_names = {mn.replace("${model}", model_base) for mn in model_names}

    if len(model_names) > 1:
        print(f"[AVISO] model.name nao E consistente entre as etapas em {script_path}: {model_names}")
        print("        Isso normalmente indica o mesmo bug que ja corrigimos (LLM_ vs LLM_V02_).")
        print("        Corrija o script antes de continuar, ou use --model-name para forcar um valor.")
    if not model_names:
        sys.exit(f"[ERRO] Nao encontrei nenhum 'model.name=' em {script_path}")

    model_name = sorted(model_names)[0]
    return model_base, model_name


def run_pipeline(dataset: str, start_fold: int, end_fold: int, repo_root: Path):
    cmd = ["bash", "run.sh", dataset, str(start_fold), str(end_fold)]
    print(f"[EXEC] {' '.join(cmd)} (cwd={repo_root})")
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        sys.exit(f"[ERRO] run.sh terminou com codigo {result.returncode}. Veja o log acima.")


def stage_artifact_path(stage: str, dataset: str, model_name: str, fold: int, repo_root: Path):
    r = repo_root / "resource"
    if stage == "sparse_retrieve":
        return r / "ranking" / f"BM25_{dataset}" / f"BM25_{dataset}_{fold}.rnk"
    if stage == "fit":
        return r / "model_checkpoint" / f"{model_name}_{dataset}_{fold}.ckpt"
    if stage == "predict":
        return r / "prediction" / f"{model_name}_{dataset}" / f"fold_{fold}"
    if stage == "eval":
        return r / "result" / f"{model_name}_{dataset}" / f"{model_name}_{dataset}_{fold}.rts"
    if stage == "fuse":
        return r / "ranking" / f"Fused_{model_name}_{dataset}" / f"Fused_{model_name}_{dataset}_{fold}.rnk"
    if stage == "aggregate":
        return r / "result" / f"Aggregated_{model_name}_{dataset}" / f"Aggregated_{model_name}_{dataset}_{fold}.rts"
    raise ValueError(stage)


def check_stage_ok(stage: str, path: Path) -> bool:
    if stage == "predict":
        # pasta precisa existir e ter pelo menos um .prd nao vazio
        return path.is_dir() and any(p.stat().st_size > 0 for p in path.glob("*.prd"))
    return path.is_file() and path.stat().st_size > 0


def run_diagnostics(dataset: str, model_name: str, folds, repo_root: Path) -> bool:
    print(f"\n=== Diagnostico: {dataset} / model.name={model_name} ===")
    all_ok = True
    for fold in folds:
        row = []
        for stage in STAGES:
            path = stage_artifact_path(stage, dataset, model_name, fold, repo_root)
            ok = check_stage_ok(stage, path)
            row.append("OK" if ok else "FALTANDO")
            if not ok:
                all_ok = False
                print(f"  [fold {fold}] {stage:15s} -> FALTANDO ({path})")
        if all(s == "OK" for s in row):
            print(f"  [fold {fold}] todas as etapas OK ({', '.join(STAGES)})")
    if all_ok:
        print("Todas as etapas de todos os folds estao presentes.\n")
    else:
        print("\nHa etapas faltando acima. Rode novamente apenas essas etapas/folds "
              "(edite o run/<dataset>.sh comentando as demais, ou rode main.py "
              "diretamente com tasks=[<etapa>] data.folds=[<fold>]).\n")
    return all_ok


def load_eval_results(dataset: str, model_name: str, folds, repo_root: Path) -> pd.DataFrame:
    rows = []
    result_dir = repo_root / "resource" / "result" / f"{model_name}_{dataset}"
    for fold in folds:
        path = result_dir / f"{model_name}_{dataset}_{fold}.rts"
        if not check_stage_ok("eval", path):
            print(f"[AVISO] pulando fold {fold}: {path} nao existe/esta vazio")
            continue
        df = pd.read_csv(path, sep="\t")
        df = df[df["split"] == "test"]
        rows.append(df)
    if not rows:
        sys.exit(f"[ERRO] Nenhum resultado de eval encontrado em {result_dir}. "
                  f"Rode o pipeline (--run) ou verifique o --dataset/--model-name.")
    return pd.concat(rows, ignore_index=True)


def summarize_ndcg_precision(eval_df: pd.DataFrame) -> dict:
    """Retorna dict {(metric, cls, k): 'media(desvio)'} para ndcg e precision."""
    out = {}
    for metric in ["ndcg", "precision"]:
        for cls in ["tail", "head"]:
            for k in THRESHOLDS:
                col = f"{metric}@{k}"
                sub = eval_df[eval_df["cls"] == cls]
                if col not in sub.columns or sub.empty:
                    out[(metric, cls, k)] = "N/A"
                    continue
                vals = sub[col].to_numpy(dtype=float) * 100
                mean, std = vals.mean(), vals.std()
                out[(metric, cls, k)] = f"{mean:.1f}({std:.1f})"
    return out


def load_relevance_map(dataset: str, repo_root: Path) -> dict:
    path = repo_root / "resource" / "dataset" / dataset / "relevance_map.pkl"
    with open(path, "rb") as f:
        raw = pickle.load(f)
    # formato cru: {text_idx: [label_idx, ...]}
    return {f"text_{text_idx}": [f"label_{l}" for l in labels_ids]
            for text_idx, labels_ids in raw.items()}


def load_all_label_ids(dataset: str, repo_root: Path):
    path = repo_root / "resource" / "dataset" / dataset / "label_cls.pkl"
    with open(path, "rb") as f:
        label_cls = pickle.load(f)
    return sorted(f"label_{l}" for l in label_cls.keys())


def compute_macro_micro_f1(dataset: str, model_name: str, folds, repo_root: Path) -> dict:
    """
    Calcula Macro-F1@k e Micro-F1@k a partir do ranking ja combinado
    (tail + head) gerado pela etapa 'aggregate'.

    OBS: RankingAggregationHelper nao grava colunas chamadas literalmente
    "Mac-F1"/"Mic-F1" -- ele grava mEtricas do ranx (f1@k no estilo
    "por consulta"). Aqui calculamos Macro-F1 e Micro-F1 "de verdade"
    (sklearn, media sobre classes / media sobre amostras) a partir do
    top-k de labels previstas vs. as labels verdadeiras, que E a
    definicao usual desses nomes na literatura de classificacao
    extrema multi-rotulo. Se o seu artigo/planilha usa uma definicao
    diferente, me avise para eu ajustar a formula.
    """
    from sklearn.metrics import f1_score

    relevance_map = load_relevance_map(dataset, repo_root)
    all_labels = load_all_label_ids(dataset, repo_root)
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit([all_labels])

    per_fold_scores = {("macro", k): [] for k in THRESHOLDS}
    per_fold_scores.update({("micro", k): [] for k in THRESHOLDS})

    ranking_dir = repo_root / "resource" / "ranking" / f"Aggregated_{model_name}_{dataset}"
    for fold in folds:
        path = ranking_dir / f"Aggregated_{model_name}_{dataset}_{fold}.rnk"
        if not check_stage_ok("aggregate", path.parent.parent / "result" if False else path):
            # fallback simples: so existencia de arquivo
            if not path.is_file():
                print(f"[AVISO] pulando fold {fold} no calculo de F1: {path} nao existe")
                continue
        with open(path, "rb") as f:
            ranking = pickle.load(f)

        text_ids = [t for t in ranking.keys() if t in relevance_map]
        if not text_ids:
            continue

        y_true = mlb.transform([relevance_map[t] for t in text_ids])

        for k in THRESHOLDS:
            pred_labels_per_text = []
            for t in text_ids:
                labels_scores = ranking[t]
                top_k = sorted(labels_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
                pred_labels_per_text.append([lbl for lbl, _ in top_k])
            y_pred = mlb.transform(pred_labels_per_text)

            per_fold_scores[("macro", k)].append(
                f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
            )
            per_fold_scores[("micro", k)].append(
                f1_score(y_true, y_pred, average="micro", zero_division=0) * 100
            )

    out = {}
    for kind in ["macro", "micro"]:
        for k in THRESHOLDS:
            vals = per_fold_scores[(kind, k)]
            if not vals:
                out[(kind, k)] = "N/A"
                continue
            vals = np.array(vals)
            out[(kind, k)] = f"{vals.mean():.1f}({vals.std():.1f})"
    return out


def build_final_table(dataset: str, model_name: str, ndcg_prec: dict, f1_scores: dict) -> pd.DataFrame:
    columns = (
        [f"nDCG_tail@{k}" for k in THRESHOLDS]
        + [f"nDCG_head@{k}" for k in THRESHOLDS]
        + [f"Precision_tail@{k}" for k in THRESHOLDS]
        + [f"Precision_head@{k}" for k in THRESHOLDS]
        + [f"Mac-F1@{k}" for k in THRESHOLDS]
        + [f"Mic-F1@{k}" for k in THRESHOLDS]
    )
    row = (
        [ndcg_prec[("ndcg", "tail", k)] for k in THRESHOLDS]
        + [ndcg_prec[("ndcg", "head", k)] for k in THRESHOLDS]
        + [ndcg_prec[("precision", "tail", k)] for k in THRESHOLDS]
        + [ndcg_prec[("precision", "head", k)] for k in THRESHOLDS]
        + [f1_scores[("macro", k)] for k in THRESHOLDS]
        + [f1_scores[("micro", k)] for k in THRESHOLDS]
    )
    return pd.DataFrame([row], columns=columns, index=[f"RAG-Fuse ({model_name}/{dataset})"])


def main():
    parser = argparse.ArgumentParser(description="Roda/agrega os resultados do pipeline RAG-Fuse.")
    parser.add_argument("--dataset", required=True, help="ex: reut90_is_lssm, webkb, acm...")
    parser.add_argument("--model-name", default=None,
                         help="Forca o model.name (ex: LLM_RetrieverRoBERTa). "
                              "Por padrao E lido de run/<dataset>.sh")
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=9)
    parser.add_argument("--run", action="store_true",
                         help="Executa o pipeline (bash run.sh <dataset> <start> <end>) antes de agregar")
    parser.add_argument("--check-only", action="store_true",
                         help="So verifica quais etapas/folds ja existem, nao agrega nada")
    parser.add_argument("--skip-diagnostics", action="store_true",
                         help="Pula a checagem de artefatos antes de agregar (nao recomendado)")
    parser.add_argument("--output", default=None, help="Caminho do CSV de saIda")
    args = parser.parse_args()

    repo_root = Path.cwd()
    if not (repo_root / "main.py").exists() or not (repo_root / "run").is_dir():
        sys.exit("[ERRO] Rode este script na raiz do repositorio RAG-Fuse "
                  "(onde ficam main.py, run/, resource/, run.sh).")

    folds = list(range(args.start_fold, args.end_fold + 1))

    model_base, model_name = parse_run_script(args.dataset, repo_root)
    if args.model_name:
        model_name = args.model_name
    print(f"[INFO] dataset={args.dataset} model_base={model_base} model.name={model_name} folds={folds}")

    if args.run:
        run_pipeline(args.dataset, args.start_fold, args.end_fold, repo_root)

    all_ok = True
    if not args.skip_diagnostics or args.check_only:
        all_ok = run_diagnostics(args.dataset, model_name, folds, repo_root)

    if args.check_only:
        sys.exit(0 if all_ok else 1)

    if not all_ok:
        print("[AVISO] Nem todas as etapas/folds estao completas. "
              "A agregacao abaixo vai usar so os folds disponIveis.\n")

    eval_df = load_eval_results(args.dataset, model_name, folds, repo_root)
    ndcg_prec = summarize_ndcg_precision(eval_df)
    f1_scores = compute_macro_micro_f1(args.dataset, model_name, folds, repo_root)

    table = build_final_table(args.dataset, model_name, ndcg_prec, f1_scores)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== Tabela final (media(desvio) entre folds, x100) ===")
    print(table.to_string())

    out_path = Path(args.output) if args.output else (
        repo_root / "resource" / "result" / f"{model_name}_{args.dataset}_final_table.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path)
    print(f"\nCSV salvo em: {out_path}")


if __name__ == "__main__":
    main()
