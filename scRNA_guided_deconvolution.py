# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr


TCGA_META_PATH = "data/TCGA_PRAD_meta_full.tsv"
TCGA_META_FALLBACK = "data/TCGA_meta_full_with_NNLS.tsv"
TCGA_COUNTS_PATH = "data/TCGA_PRAD_counts_tumor_barcode.tsv"
REF_SIGNATURE_PATH = "data/ref_signature.tsv"
OUT_DIR = "results/scRNA_TCGA"

BULK_NORM_TARGET = 1e6
MIN_SHARED_GENES = 300
USE_GENE_SCALING = True
NORMALIZE_COEF = True
ENTROPY_EPS = 1e-12


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def save_tsv(df, path, index=False):
    ensure_dir(Path(path).parent)
    df.to_csv(path, sep="\t", index=index)


def gene_symbol(x):
    x = str(x).strip()
    if "|" in x:
        x = x.split("|")[-1]
    if "." in x:
        parts = x.split(".")
        if len(parts) == 2 and parts[1].isdigit():
            x = parts[0]
    return x.upper()


def load_tcga_meta():
    if Path(TCGA_META_PATH).exists():
        meta = read_tsv(TCGA_META_PATH)
    elif Path(TCGA_META_FALLBACK).exists():
        meta = read_tsv(TCGA_META_FALLBACK)
    else:
        raise FileNotFoundError("TCGA meta file not found")

    if "Sample_UID" not in meta.columns:
        raise ValueError("Sample_UID is required in TCGA meta")
    if "risk_score_mu" not in meta.columns:
        raise ValueError("risk_score_mu is required in TCGA meta")

    if "Sample_ID" not in meta.columns:
        meta["Sample_ID"] = (
            meta["Sample_UID"]
            .astype(str)
            .str.split("__", n=1)
            .str[0]
            .str[:12]
        )

    meta["Sample_ID"] = meta["Sample_ID"].astype(str).str[:12]
    meta["risk_score_mu"] = pd.to_numeric(meta["risk_score_mu"], errors="coerce")
    meta = meta.dropna(subset=["Sample_UID", "Sample_ID", "risk_score_mu"]).copy()
    return meta


def load_bulk_counts():
    if not Path(TCGA_COUNTS_PATH).exists():
        raise FileNotFoundError(f"Bulk counts file not found: {TCGA_COUNTS_PATH}")

    df = pd.read_csv(TCGA_COUNTS_PATH, sep="\t", low_memory=False)
    gene_col = df.columns[0]

    genes = df[gene_col].astype(str).map(gene_symbol)
    mat = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    mat.index = genes
    mat = mat.groupby(mat.index).mean()
    return mat


def align_bulk_to_meta(meta, bulk_counts):
    bulk_cols = pd.Index([str(x) for x in bulk_counts.columns])
    bulk_id12 = pd.Index([x[:12] for x in bulk_cols])

    id_to_uid = (
        meta[["Sample_ID", "Sample_UID"]]
        .drop_duplicates(subset=["Sample_ID"])
        .set_index("Sample_ID")["Sample_UID"]
        .to_dict()
    )

    keep_ids = [sid for sid in meta["Sample_ID"].tolist() if sid in set(bulk_id12)]
    if len(keep_ids) == 0:
        raise ValueError("No overlap between bulk columns and meta Sample_ID")

    chosen_cols = []
    chosen_uid = []
    used_idx = set()

    for sid in keep_ids:
        idx = np.where(bulk_id12 == sid)[0]
        idx = [i for i in idx if i not in used_idx]
        if len(idx) == 0:
            continue
        i = idx[0]
        used_idx.add(i)
        chosen_cols.append(bulk_cols[i])
        chosen_uid.append(id_to_uid[sid])

    bulk = bulk_counts.loc[:, chosen_cols].copy()
    bulk.columns = chosen_uid

    meta2 = meta.loc[meta["Sample_UID"].isin(chosen_uid)].copy()
    meta2 = meta2.drop_duplicates(subset=["Sample_UID"]).copy()
    meta2 = meta2.set_index("Sample_UID").loc[chosen_uid].reset_index()

    return meta2, bulk


def logcpm(counts):
    x = counts.to_numpy(dtype=float, copy=False)
    lib = np.nansum(x, axis=0)
    lib[lib <= 0] = np.nan
    cpm = x / lib[None, :] * BULK_NORM_TARGET
    cpm = np.log1p(cpm)
    return pd.DataFrame(cpm, index=counts.index, columns=counts.columns)


def load_ref_signature():
    if not Path(REF_SIGNATURE_PATH).exists():
        raise FileNotFoundError(f"Reference signature file not found: {REF_SIGNATURE_PATH}")

    sig = read_tsv(REF_SIGNATURE_PATH)
    gene_col = sig.columns[0]
    sig = sig.rename(columns={gene_col: "gene"})
    sig["gene"] = sig["gene"].astype(str).map(gene_symbol)
    sig = sig.drop_duplicates(subset=["gene"]).set_index("gene")

    for c in sig.columns:
        sig[c] = pd.to_numeric(sig[c], errors="coerce").fillna(0.0)

    return sig


def build_hybrid_signature(sig_fine):
    epi_ar = ["Epi_Luminal_AR_high"]
    epi_pro = ["Epi_Proliferating"]
    epi_plastic = ["Epi_Other"]

    immune_tnk = ["T.CD4", "T.CD8", "T.proliferating", "NK"]
    immune_bplasma = ["B", "Plasma"]
    immune_myeloid = ["Monocyte", "Macrophage"]

    stromal = ["Fibroblast", "Endothelial", "SM"]
    mast = ["Mast"]

    groups = {
        "Epi_Luminal_AR_high": epi_ar,
        "Epi_Proliferating": epi_pro,
        "Epi_Plasticity": epi_plastic,
        "Immune_TNK": immune_tnk,
        "Immune_BPlasma": immune_bplasma,
        "Immune_Myeloid": immune_myeloid,
        "Stromal": stromal,
        "Mast": mast,
    }

    cols = set(sig_fine.columns)
    missing = {}
    for k, v in groups.items():
        miss = [x for x in v if x not in cols]
        if len(miss) > 0:
            missing[k] = miss
    if len(missing) > 0:
        raise ValueError(f"Missing columns in ref_signature.tsv: {missing}")

    hyb = pd.DataFrame(index=sig_fine.index)
    for k, v in groups.items():
        hyb[k] = sig_fine[v].mean(axis=1)

    mapping_rows = []
    for k, v in groups.items():
        for x in v:
            mapping_rows.append({
                "hybrid_state": k,
                "reference_state": x,
            })
    mapping_df = pd.DataFrame(mapping_rows)

    return hyb, mapping_df


def run_nnls_deconvolution(bulk_expr, ref_sig):
    common = bulk_expr.index.intersection(ref_sig.index)
    if len(common) < MIN_SHARED_GENES:
        raise ValueError(f"Too few shared genes: {len(common)}")

    Y = bulk_expr.loc[common].to_numpy(dtype=float, copy=False)
    A = ref_sig.loc[common].to_numpy(dtype=float, copy=False)

    states = ref_sig.columns.tolist()
    samples = bulk_expr.columns.tolist()

    if USE_GENE_SCALING:
        gene_sd = np.nanstd(A, axis=1, ddof=0)
        gene_sd[~np.isfinite(gene_sd)] = 0.0
        gene_sd[gene_sd <= 1e-8] = 1.0
        A_use = A / gene_sd[:, None]
    else:
        gene_sd = np.ones(A.shape[0], dtype=float)
        A_use = A

    coef = np.full((len(samples), len(states)), np.nan, dtype=float)
    diag_rows = []

    for j, sid in enumerate(samples):
        y = Y[:, j].copy()
        ok = np.isfinite(y) & np.isfinite(A_use).all(axis=1)

        n_shared = int(ok.sum())
        if n_shared < MIN_SHARED_GENES:
            diag_rows.append({
                "Sample_UID": sid,
                "n_shared_genes": n_shared,
                "residual_ss": np.nan,
                "valid_nnls": False,
            })
            continue

        A_j = A_use[ok]
        y_j = y[ok] / gene_sd[ok]

        q, resid = nnls(A_j, y_j)

        if NORMALIZE_COEF:
            s = q.sum()
            if np.isfinite(s) and s > 0:
                q = q / s

        coef[j, :] = q
        diag_rows.append({
            "Sample_UID": sid,
            "n_shared_genes": n_shared,
            "residual_ss": float(resid),
            "valid_nnls": True,
        })

    frac = pd.DataFrame(coef, index=samples, columns=states).reset_index()
    frac = frac.rename(columns={"index": "Sample_UID"})

    diag = pd.DataFrame(diag_rows)

    x = frac.drop(columns="Sample_UID").to_numpy(dtype=float)

    row_sum = np.nansum(x, axis=1)

    max_frac = np.full(x.shape[0], np.nan, dtype=float)
    for i in range(x.shape[0]):
        row = x[i]
        if np.isfinite(row).any():
            max_frac[i] = np.nanmax(row)

    p = np.where(np.isfinite(x), x, 0.0)
    rs = p.sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    p = p / rs

    entropy = -np.sum(p * np.log(p + ENTROPY_EPS), axis=1)
    if p.shape[1] > 1:
        entropy_norm = entropy / np.log(p.shape[1])
    else:
        entropy_norm = entropy.copy()

    diag["row_sum"] = row_sum
    diag["max_frac"] = max_frac
    diag["entropy"] = entropy
    diag["entropy_norm"] = entropy_norm

    return frac, diag


def association_with_mu(meta, frac):
    df = meta[["Sample_UID", "risk_score_mu"]].merge(frac, on="Sample_UID", how="inner")

    results = []
    frac_cols = [c for c in frac.columns if c != "Sample_UID"]

    for col in frac_cols:
        tmp = df[["risk_score_mu", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna()

        if tmp.shape[0] < 5:
            results.append({
                "state": col,
                "n": int(tmp.shape[0]),
                "rho": np.nan,
                "p_value": np.nan,
            })
            continue

        rho, p = spearmanr(tmp["risk_score_mu"], tmp[col], nan_policy="omit")
        results.append({
            "state": col,
            "n": int(tmp.shape[0]),
            "rho": float(rho),
            "p_value": float(p),
        })

    out = pd.DataFrame(results)
    out = out.sort_values(["p_value", "state"], na_position="last").reset_index(drop=True)
    return out


def main():
    ensure_dir(OUT_DIR)

    meta = load_tcga_meta()
    bulk_counts = load_bulk_counts()
    meta_used, bulk_used = align_bulk_to_meta(meta, bulk_counts)

    bulk_expr = logcpm(bulk_used)
    ref_sig = load_ref_signature()
    hybrid_sig, hybrid_map = build_hybrid_signature(ref_sig)

    frac, diag = run_nnls_deconvolution(bulk_expr, hybrid_sig)
    assoc = association_with_mu(meta_used, frac)

    save_tsv(meta_used, f"{OUT_DIR}/TCGA_meta_used.tsv")
    save_tsv(
        hybrid_sig.reset_index().rename(columns={"index": "gene"}),
        f"{OUT_DIR}/ref_signature_HYBRID.tsv"
    )
    save_tsv(hybrid_map, f"{OUT_DIR}/HYBRID_signature_mapping.tsv")
    save_tsv(frac, f"{OUT_DIR}/HYBRID_fractions.tsv")
    save_tsv(diag, f"{OUT_DIR}/HYBRID_fraction_diagnostics.tsv")
    save_tsv(assoc, f"{OUT_DIR}/Table_HYBRID_association_mu.tsv")

    print("[OK] scRNA-guided deconvolution finished")
    print(f"[OK] samples used: {meta_used.shape[0]}")
    print(f"[OK] bulk genes: {bulk_expr.shape[0]}")
    print(f"[OK] hybrid states: {hybrid_sig.shape[1]}")


if __name__ == "__main__":
    main()