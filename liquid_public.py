# -*- coding: utf-8 -*-

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MOESM3_XLSX = "data/liquid/43018_2023_692_MOESM3_ESM.xlsx"
PROJECTOR_DIR = "data/cohort"
OUT_DIR = "results/liquid"

RB_AXIS = ["RB1", "PTEN", "TP53"]
CONTEXT_GAINS = ["AR", "MYC"]
DRIVERS = RB_AXIS + CONTEXT_GAINS

BASELINE_MODE = "status_then_maxTF"
TF_STRICT = 2.0


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def save_tsv(df, path):
    ensure_dir(Path(path).parent)
    df.to_csv(path, sep="\t", index=False)


def first_existing(root, names):
    for name in names:
        fp = Path(root) / name
        if fp.exists():
            return fp
    raise FileNotFoundError(f"Cannot find any of: {names}")


def mad(x):
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def baseline_rank(status):
    if pd.isna(status):
        return 99
    s = str(status)

    if "mCSPC PreADT PreRP" in s or "mHSPC PreADT PreRP" in s:
        return 1
    if "mCSPC PostADT PreRP" in s or "mHSPC PostADT PreRP" in s:
        return 2
    if "mCSPC PreADT PostRP" in s or "mHSPC PreADT PostRP" in s:
        return 3
    if "mCSPC PostADT PostRP" in s or "mHSPC PostADT PostRP" in s:
        return 4
    if "mCRPC" in s:
        return 10
    return 50


def parse_isup_group(x):
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass

    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass

    s = str(x).strip().lower()
    if s in ("", "nan", "none", "[]"):
        return np.nan
    if "1-3" in s or "≤ 7" in s or "<= 7" in s:
        return 2.0
    if "4-5" in s or "≥ 8" in s or ">= 8" in s:
        return 4.5

    m = re.search(r"grade\s*group\s*([1-5])", s)
    if m:
        return float(m.group(1))

    m = re.search(r"([1-5])", s)
    if m:
        return float(m.group(1))

    return np.nan


def parse_t_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).upper().strip().replace("PT", "T").replace(" ", "")
    m = re.search(r"([1-4])", s)
    return float(m.group(1)) if m else np.nan


def parse_n_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).upper().strip().replace("PN", "N").replace(" ", "")
    if "N0" in s:
        return 0.0
    if "N1" in s:
        return 1.0

    v = to_num(pd.Series([x])).iloc[0]
    if np.isfinite(v) and v in (0, 1):
        return float(v)
    return np.nan


def read_sheet(xlsx, name):
    last_err = None
    for h in (1, 0):
        try:
            return pd.read_excel(xlsx, sheet_name=name, header=h)
        except Exception as e:
            last_err = e
    raise last_err


def load_projector(projector_dir):
    beta_path = first_existing(projector_dir, ["projector_beta.tsv"])
    scaler_path = first_existing(projector_dir, ["projector_scaler.tsv", "projector_scaler.json"])
    aux_path = first_existing(projector_dir, ["projector_aux.tsv", "projector_aux.json"])

    beta = pd.read_csv(beta_path, sep="\t")
    if not {"term", "coef"}.issubset(beta.columns):
        raise ValueError("projector_beta.tsv must contain term and coef")

    if str(scaler_path).endswith(".tsv"):
        scaler = pd.read_csv(scaler_path, sep="\t")
        age_mean = float(scaler.loc[scaler["term"] == "age_mean", "value"].iloc[0])
        age_sd = float(scaler.loc[scaler["term"] == "age_sd", "value"].iloc[0])
    else:
        import json
        with open(scaler_path, "r", encoding="utf-8") as f:
            scaler = json.load(f)
        age_mean = float(scaler["age_mean"])
        age_sd = float(scaler["age_sd"])

    intercept = 0.0
    if (beta["term"].astype(str).str.lower() == "intercept").any():
        intercept = float(beta.loc[beta["term"].astype(str).str.lower() == "intercept", "coef"].iloc[0])

    coef = {}
    for _, r in beta.iterrows():
        term = str(r["term"]).strip()
        if term.lower() == "intercept":
            continue
        coef[term] = float(r["coef"])

    return {
        "intercept": intercept,
        "coef": coef,
        "age_mean": age_mean,
        "age_sd": age_sd,
        "beta_path": str(beta_path),
        "scaler_path": str(scaler_path),
        "aux_path": str(aux_path),
    }


def compute_mu(df, projector):
    age = to_num(df["age"])
    age_z = (age - projector["age_mean"]) / projector["age_sd"]

    mu = np.full(len(df), float(projector["intercept"]), dtype=float)

    for term, b in projector["coef"].items():
        if term == "age":
            x = age_z.fillna(0.0).astype(float).to_numpy()
        elif term == "ISUP":
            x = to_num(df["ISUP_num"]).fillna(0.0).astype(float).to_numpy()
        elif term == "T_num":
            x = to_num(df["T_num"]).fillna(0.0).astype(float).to_numpy()
        elif term == "N_num":
            x = to_num(df["N_num"]).fillna(0.0).astype(float).to_numpy()
        else:
            continue
        mu += float(b) * x

    return pd.Series(mu, index=df.index, name="mu")


def load_clinical(xlsx):
    s1 = read_sheet(xlsx, "S1")

    need = [
        "Patient ID",
        "Age at diagnosis",
        "pT-stage",
        "pN-stage",
        "ISUP grade group (biopsy)",
        "ISUP grade group (prostatectomy)",
        "mCRPC progression",
        "Time to CRPC (months)",
    ]
    miss = [c for c in need if c not in s1.columns]
    if miss:
        raise ValueError(f"S1 missing columns: {miss}")

    clin = s1[need].copy().rename(columns={"Patient ID": "patient_id"})
    clin["event"] = clin["mCRPC progression"].map({"Yes": 1, "No": 0})
    clin["ttcr_months"] = to_num(clin["Time to CRPC (months)"])
    clin["age"] = to_num(clin["Age at diagnosis"])

    clin["ISUP_num"] = clin["ISUP grade group (biopsy)"].apply(parse_isup_group)
    clin["ISUP_num"] = clin["ISUP_num"].where(
        clin["ISUP_num"].notna(),
        clin["ISUP grade group (prostatectomy)"].apply(parse_isup_group),
    )
    clin["ISUP_num"] = to_num(clin["ISUP_num"])

    clin["T_num"] = clin["pT-stage"].apply(parse_t_num)
    clin["N_num"] = clin["pN-stage"].apply(parse_n_num)

    return clin


def select_baseline_cfDNA(xlsx):
    s2 = read_sheet(xlsx, "S2").rename(columns={"Patient ID": "patient_id"}).copy()
    s3 = read_sheet(xlsx, "S3")
    s4 = read_sheet(xlsx, "S4")

    cf = s2[s2["Sample category"] == "cfDNA"].copy()
    cf = cf.merge(
        s3[["Sample ID", "QC", "DNA yield (ng)", "Targeted-sequencing coverage (median)"]],
        on="Sample ID", how="left"
    )
    cf = cf.merge(
        s4[["Sample ID", "Tumor fraction (%)"]],
        on="Sample ID", how="left"
    )

    cf["qc_pass"] = cf["QC"].isna()
    cf["baseline_rank"] = cf["Patient status at collection"].apply(baseline_rank)
    cf["TF"] = to_num(cf["Tumor fraction (%)"]).fillna(0.0)
    cf["coverage_median"] = to_num(cf["Targeted-sequencing coverage (median)"])
    cf["cfDNA_idx"] = to_num(cf["Sample ID"].astype(str).str.extract(r"cfDNA_(\d+)", expand=False))

    if BASELINE_MODE == "status_then_maxTF":
        sort_cols = ["patient_id", "baseline_rank", "qc_pass", "TF", "coverage_median", "cfDNA_idx"]
        asc = [True, True, False, False, False, True]
    else:
        sort_cols = ["patient_id", "baseline_rank", "qc_pass", "coverage_median", "cfDNA_idx"]
        asc = [True, True, False, False, True]

    baseline = (
        cf.sort_values(sort_cols, ascending=asc)
          .groupby("patient_id", as_index=False)
          .first()
          .rename(columns={"Sample ID": "cfDNA_sample_id"})
    )

    return baseline


def select_wbc_pair(xlsx):
    s2 = read_sheet(xlsx, "S2").rename(columns={"Patient ID": "patient_id"}).copy()
    s3 = read_sheet(xlsx, "S3")

    wbc = s2[s2["Sample category"] == "WBC"].copy()
    wbc = wbc.merge(
        s3[["Sample ID", "QC", "Targeted-sequencing coverage (median)"]],
        on="Sample ID", how="left"
    )

    wbc["qc_pass_wbc"] = wbc["QC"].isna()
    wbc["coverage_median_wbc"] = to_num(wbc["Targeted-sequencing coverage (median)"])

    wbc_sel = (
        wbc.sort_values(["patient_id", "qc_pass_wbc", "coverage_median_wbc"], ascending=[True, False, False])
           .groupby("patient_id", as_index=False)
           .first()
           .rename(columns={"Sample ID": "WBC_sample_id"})
    )
    return wbc_sel


def build_cn_table(xlsx, manifest):
    s7 = read_sheet(xlsx, "S7").rename(columns={"Patient ID": "patient_id"}).copy()

    need = ["patient_id", "Sample ID", "GENE", "Log_ratio"]
    miss = [c for c in need if c not in s7.columns]
    if miss:
        raise ValueError(f"S7 missing columns: {miss}")

    base_ids = manifest["cfDNA_sample_id"].dropna().unique().tolist()
    s7b = s7[s7["Sample ID"].isin(base_ids)].copy()

    var_tbl = (
        s7b.groupby("Sample ID")["Log_ratio"]
           .agg(
               CN_MAD_allgenes=lambda x: mad(np.asarray(x, dtype=float)),
               CN_abs95_allgenes=lambda x: float(np.quantile(np.abs(np.asarray(x, dtype=float)), 0.95)),
               CN_SD_allgenes=lambda x: float(np.std(np.asarray(x, dtype=float))),
           )
           .reset_index()
           .rename(columns={"Sample ID": "cfDNA_sample_id"})
    )

    drv = s7b[s7b["GENE"].isin(DRIVERS)].copy()

    wide = (
        drv.pivot_table(index=["patient_id", "Sample ID"], columns="GENE", values="Log_ratio", aggfunc="first")
           .reset_index()
           .rename(columns={"Sample ID": "cfDNA_sample_id"})
    )
    wide.columns.name = None

    for g in DRIVERS:
        if g not in wide.columns:
            wide[g] = np.nan

    wide["Loss_score"] = wide[RB_AXIS].min(axis=1)
    wide["Gain_score"] = wide[CONTEXT_GAINS].max(axis=1)
    wide["CN_absmax"] = wide[DRIVERS].abs().max(axis=1)

    cn = (
        wide.merge(var_tbl, on="cfDNA_sample_id", how="left")
            .merge(
                manifest[[
                    "patient_id", "cfDNA_sample_id", "WBC_sample_id", "has_WBC",
                    "baseline_rank", "qc_pass", "TF", "coverage_median"
                ]],
                on=["patient_id", "cfDNA_sample_id"], how="left"
            )
    )

    return cn


def add_rb_alt_scores(df):
    out = df.copy()
    M = out[RB_AXIS].apply(pd.to_numeric, errors="coerce")
    out["Loss_mean"] = M.mean(axis=1)
    return out


def spearman_safe(a, b):
    x = to_num(a).to_numpy()
    y = to_num(b).to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 8:
        return {"rho": np.nan, "p_value": np.nan, "n": n}
    rho, p = spearmanr(x[m], y[m])
    return {"rho": float(rho), "p_value": float(p), "n": n}


def build_subsets(dt):
    thr_abs95 = float(np.nanmedian(to_num(dt["CN_abs95_allgenes"])))

    out = dt.copy()
    out["TF_gt0"] = out["TF"] > 0
    out["TF_ge2"] = out["TF"] >= TF_STRICT
    out["detectable_cn"] = to_num(out["CN_abs95_allgenes"]) >= thr_abs95
    out["detectable_any"] = out["TF_gt0"] | out["detectable_cn"]

    subsets = {
        "all": out,
        "detectable_any": out[out["detectable_any"]],
        "detectable_any_mHSPC": out[(out["detectable_any"]) & (out["baseline_rank"] <= 4)],
        f"TF_ge{int(TF_STRICT)}": out[out["TF_ge2"]],
        f"TF_ge{int(TF_STRICT)}_mHSPC": out[(out["TF_ge2"]) & (out["baseline_rank"] <= 4)],
    }

    return out, subsets, thr_abs95


def association_table(subsets):
    report_vars = [
        "RB1", "PTEN", "TP53",
        "Loss_score", "Loss_mean",
        "CN_abs95_allgenes", "CN_MAD_allgenes", "CN_absmax",
        "AR", "MYC", "Gain_score",
    ]

    rows = []
    for sname, sub in subsets.items():
        for v in report_vars:
            if v in sub.columns:
                s = spearman_safe(sub["mu"], sub[v])
                rows.append({"subset": sname, "variable": v, **s})

    return pd.DataFrame(rows)


def main():
    ensure_dir(OUT_DIR)

    projector = load_projector(PROJECTOR_DIR)
    clinical = load_clinical(MOESM3_XLSX)
    baseline = select_baseline_cfDNA(MOESM3_XLSX)
    wbc = select_wbc_pair(MOESM3_XLSX)

    manifest = baseline.merge(
        wbc[["patient_id", "WBC_sample_id", "qc_pass_wbc", "coverage_median_wbc"]],
        on="patient_id",
        how="left"
    )
    manifest["has_WBC"] = manifest["WBC_sample_id"].notna()

    cn = build_cn_table(MOESM3_XLSX, manifest)

    dt = cn.merge(
        clinical[["patient_id", "age", "ISUP_num", "T_num", "N_num", "event", "ttcr_months"]],
        on="patient_id",
        how="left"
    )

    dt["mu"] = compute_mu(dt, projector)
    dt["mu_tertile"] = pd.qcut(dt["mu"], 3, labels=["low", "mid", "high"], duplicates="drop")

    dt = add_rb_alt_scores(dt)
    dt, subsets, thr_abs95 = build_subsets(dt)

    assoc = association_table(subsets)

    save_tsv(clinical, f"{OUT_DIR}/clinical_used.tsv")
    save_tsv(manifest, f"{OUT_DIR}/baseline_manifest.tsv")
    save_tsv(cn, f"{OUT_DIR}/cn_driver_scores.tsv")
    save_tsv(dt, f"{OUT_DIR}/liquid_master.tsv")
    save_tsv(assoc, f"{OUT_DIR}/spearman_mu_cn.tsv")

    summary = pd.DataFrame([{
        "n_total": int(dt.shape[0]),
        "n_detectable_any": int(dt["detectable_any"].sum()),
        "n_detectable_any_mHSPC": int(((dt["detectable_any"]) & (dt["baseline_rank"] <= 4)).sum()),
        "thr_abs95_median": float(thr_abs95),
        "projector_beta": projector["beta_path"],
        "projector_scaler": projector["scaler_path"],
    }])
    save_tsv(summary, f"{OUT_DIR}/summary.tsv")

    print("[OK] liquid core analysis finished")
    print(f"[OK] patients used: {dt.shape[0]}")
    print(f"[OK] detectable_any: {int(dt['detectable_any'].sum())}")
    print(f"[OK] output: {OUT_DIR}")


if __name__ == "__main__":
    main()