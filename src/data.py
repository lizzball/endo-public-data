from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests

GEO_ACCESSION = "GSE279435"
SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE279nnn/"
    "GSE279435/matrix/GSE279435_series_matrix.txt.gz"
)


def download_series_matrix(cache_path: str | Path = "data/GSE279435_series_matrix.txt.gz") -> Path:
    """Download the GEO series matrix if it is not already cached locally."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    response = requests.get(SERIES_MATRIX_URL, timeout=120)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return cache_path


def _read_matrix_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw).decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _clean_token(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def parse_series_matrix(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse GEO series matrix into sample x feature data and metadata."""
    text = _read_matrix_text(path)
    lines = text.splitlines()

    repeated_meta: Dict[str, List[List[str]]] = {}
    table_start = None
    table_end = None

    for i, line in enumerate(lines):
        if line == "!series_matrix_table_begin":
            table_start = i + 1
            continue
        if line == "!series_matrix_table_end":
            table_end = i
            break
        if line.startswith("!Sample_"):
            parts = line.split("\t")
            key = parts[0]
            values = [_clean_token(v) for v in parts[1:]]
            repeated_meta.setdefault(key, []).append(values)

    if table_start is None or table_end is None:
        raise ValueError("Could not locate the GEO series-matrix expression table.")

    table_text = "\n".join(lines[table_start:table_end])
    expression = pd.read_csv(
        io.StringIO(table_text),
        sep="\t",
        quotechar='"',
        na_values=["null", "NULL", "NA", ""],
        low_memory=False,
    )
    if "ID_REF" not in expression.columns:
        raise ValueError("Series matrix did not contain an ID_REF column.")

    X = expression.set_index("ID_REF").T
    X = X.apply(pd.to_numeric, errors="coerce")
    X.index.name = "geo_accession"

    sample_ids = list(X.index)
    n = len(sample_ids)

    def first_row(key: str, default: str = "") -> List[str]:
        rows = repeated_meta.get(key, [])
        if not rows:
            return [default] * n
        row = rows[0]
        return (row + [default] * n)[:n]

    titles = first_row("!Sample_title")
    sources = first_row("!Sample_source_name_ch1")

    characteristics_rows = repeated_meta.get("!Sample_characteristics_ch1", [])
    characteristics = [[] for _ in range(n)]
    for row in characteristics_rows:
        for idx, value in enumerate((row + [""] * n)[:n]):
            if value:
                characteristics[idx].append(value)

    disease = []
    for vals in characteristics:
        disease_value = ""
        for value in vals:
            if value.lower().startswith("disease:"):
                disease_value = value.split(":", 1)[1].strip()
                break
        disease.append(disease_value)

    group = []
    for title, disease_value in zip(titles, disease):
        upper = title.upper().strip()
        if upper.startswith("END") or "ENDOMETRIOSIS" in disease_value.upper():
            group.append("Endometriosis")
        elif upper.startswith("CTR"):
            group.append("Control")
        else:
            group.append("Unknown")

    meta = pd.DataFrame(
        {
            "geo_accession": sample_ids,
            "title": titles,
            "source": sources,
            "disease": disease,
            "group": group,
            "characteristics": [" | ".join(v) for v in characteristics],
        }
    ).set_index("geo_accession")

    keep = meta["group"].isin(["Endometriosis", "Control"])
    meta = meta.loc[keep]
    X = X.loc[meta.index]

    counts = meta["group"].value_counts().to_dict()
    if len(meta) < 100 or counts.get("Endometriosis", 0) < 40 or counts.get("Control", 0) < 40:
        raise ValueError(
            f"Unexpected cohort after parsing: n={len(meta)}, counts={counts}. "
            "Check whether GEO metadata formatting changed."
        )

    return X, meta


def load_geo_dataset(cache_path: str | Path = "data/GSE279435_series_matrix.txt.gz"):
    path = download_series_matrix(cache_path)
    X, meta = parse_series_matrix(path)
    y = (meta["group"] == "Endometriosis").astype(int)
    return X, y, meta
