#!/usr/bin/env python3
"""
generate_run_scripts.py

Automatiza a criação dos scripts de "instance selection" em run/ para todos
os datasets, usando como referência o conjunto de variações que já existe
para acm / ohsumed / reut90:

    <dataset>.sh
    <dataset>_is_bio_b25_t50.sh
    <dataset>_is_cnn.sh
    <dataset>_is_e2sc_iterative.sh
    <dataset>_is_e2sc_r20.sh
    <dataset>_is_e2sc_r25.sh
    <dataset>_is_e2sc_r30.sh
    <dataset>_is_lssm.sh

Como funciona
-------------
1. Detecta os scripts "base" em run/ (ex.: 20ng.sh, books.sh, dblp.sh...),
   ou seja, os que não têm sufixo "_is_" no nome.
2. Para cada dataset base, garante que exista um script para cada uma das
   variações canônicas listadas acima (CANONICAL_SUFFIXES).
3. Cada script de variação é gerado como cópia exata do script base do
   próprio dataset, alterando apenas a linha:

       data=<dataset>
   para:
       data=<dataset>_is_<sufixo>

   (única diferença observada entre um script base e suas variantes já
   existentes no repositório).
4. Scripts que já existem em run/ são preservados (não são sobrescritos),
   a menos que --overwrite seja usado.

Uso
---
    # Rodar a partir da raiz do repositório (onde ficam run/ e setting/):
    python generate_run_scripts.py

    # Ver o que seria criado, sem escrever nada:
    python generate_run_scripts.py --dry-run

    # Sobrescrever variações que já existirem:
    python generate_run_scripts.py --overwrite

    # Restringir a um ou mais datasets específicos:
    python generate_run_scripts.py --datasets 20ng books
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RUN_DIR = Path("run")

# Conjunto canônico de variações, na ordem observada em acm/ohsumed/reut90.
CANONICAL_SUFFIXES = [
    "is_bio_b25_t50",
    "is_cnn",
    "is_e2sc_iterative",
    "is_e2sc_r20",
    "is_e2sc_r25",
    "is_e2sc_r30",
    "is_lssm",
]

# Linha que identifica o dataset dentro do script, ex.: "data=acm"
DATA_LINE_RE = re.compile(r"^data=(?P<name>\S+)\s*$", re.MULTILINE)

# Nome de variante: <dataset>_is_<algo>
VARIANT_RE = re.compile(r"^(?P<base>.+?)_is_.+$")


def find_base_scripts(run_dir: Path) -> dict[str, Path]:
    """Retorna {nome_dataset: caminho_do_script} apenas para scripts
    'principais' (aqueles que não são variantes '_is_...')."""
    bases: dict[str, Path] = {}
    for sh_file in sorted(run_dir.glob("*.sh")):
        stem = sh_file.stem
        if VARIANT_RE.match(stem):
            continue  # é uma variante, não um script base
        bases[stem] = sh_file
    return bases


def build_variant_content(base_content: str, base_dataset: str, variant_name: str) -> str:
    """Substitui a linha 'data=<base_dataset>' por 'data=<variant_name>'."""

    def _replace(match: re.Match) -> str:
        if match.group("name") == base_dataset:
            return f"data={variant_name}"
        return match.group(0)

    new_content, n_subs = DATA_LINE_RE.subn(_replace, base_content, count=1)
    if n_subs == 0:
        raise ValueError(
            f"Não encontrei a linha 'data={base_dataset}' no script base. "
            "Verifique manualmente o formato do arquivo."
        )
    return new_content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir", type=Path, default=RUN_DIR, help="Pasta com os scripts .sh (padrão: run/)"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Restringe a geração a estes datasets base (ex.: 20ng books). Padrão: todos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve scripts de variante que já existam.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas mostra o que seria criado, sem escrever arquivos.",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir

    if not run_dir.is_dir():
        print(f"[erro] Pasta de scripts não encontrada: {run_dir}", file=sys.stderr)
        return 1

    base_scripts = find_base_scripts(run_dir)

    if args.datasets:
        wanted = set(args.datasets)
        missing = wanted - base_scripts.keys()
        if missing:
            print(f"[erro] Dataset(s) sem script base em {run_dir}: {sorted(missing)}", file=sys.stderr)
            return 1
        base_scripts = {k: v for k, v in base_scripts.items() if k in wanted}

    total_created = 0
    total_skipped = 0

    for dataset, base_path in base_scripts.items():
        base_content = base_path.read_text(encoding="utf-8")

        print(f"\n== {dataset} (base: {base_path.name}) ==")
        for suffix in CANONICAL_SUFFIXES:
            variant_name = f"{dataset}_{suffix}"
            target_path = run_dir / f"{variant_name}.sh"

            if target_path.exists() and not args.overwrite:
                print(f"  [pular]  {target_path.name} já existe")
                total_skipped += 1
                continue

            try:
                new_content = build_variant_content(base_content, dataset, variant_name)
            except ValueError as exc:
                print(f"  [erro]   {variant_name}: {exc}", file=sys.stderr)
                continue

            action = "sobrescrever" if target_path.exists() else "criar"
            print(f"  [{action}] {target_path.name}")

            if not args.dry_run:
                target_path.write_text(new_content, encoding="utf-8")
                total_created += 1

    print()
    if args.dry_run:
        print("(dry-run: nenhum arquivo foi escrito)")
    else:
        print(f"Concluído. {total_created} script(s) criado(s)/atualizado(s), {total_skipped} já existiam.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
