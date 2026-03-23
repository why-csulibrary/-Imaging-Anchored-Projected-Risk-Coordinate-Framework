# -*- coding: utf-8 -*-


from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm

from lifelines import CoxPHFitter


# ============================================================
# Settings
# ============================================================
@dataclass
class Settings:
    input_path: str = "data/cohort.csv"
    output_dir: str = "results"

    event_col_candidates: Tuple[str, ...] = ("Event_Status", "event", "status", "TTCR_status")
    time_col_candidates: Tuple[str, ...] = ("Event_Time", "TTCR_time", "TTCR", "time")

    age_col_candidates: Tuple[str, ...] = ("age", "Age")
    isup_col_candidates: Tuple[str, ...] = ("ISUP", "isup", "grade_group", "GG")
    t_col_candidates: Tuple[str, ...] = ("T_num", "T", "t", "clinical_T", "cT")
    n_col_candidates: Tuple[str, ...] = ("N_num", "N", "n", "clinical_N", "cN")

    prostate_dims: Tuple[str, str, str] = ("prostate1", "prostate2", "prostate3")
    suv_primary_preferred: str = "diagnose_SUVmax.1"
    suv_prefix: str = "diagnose_SUVmax."

    v_cr: float = 5000.0
    eps_sigma: float = 1e-6
    maxiter: int = 5000
    seed: int = 20260118


S = Settings()


# ============================================================
# Small utilities
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def try_read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def pick_col(df: pd.DataFrame, candidates: Tuple[str, ...], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Missing required columns: {candidates}")
    return None


def standardize_event_indicator(series: pd.Series) -> pd.Series:
    """
    Convert event indicator to binary:
    event = 1, censor = 0

    Supports:
    - {0,1}
    - {1,2}, where 1=event and 2=non-event/censor
    """
    x = pd.to_numeric(series, errors="coerce")
    observed = set(pd.Series(x.dropna().unique()).astype(int).tolist())

    if observed.issubset({0, 1}):
        return (x == 1).fillna(0).astype(int)

    if observed.issubset({1, 2}):
        return (x == 1).fillna(0).astype(int)

    return (x == 1).fillna(0).astype(int)


def zfit(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (x - mu) / sd, mu, sd


def parse_T_to_num(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip().upper()
    s = s.replace("CLINICAL", "").replace("C", "")

    for digit in ("1", "2", "3", "4"):
        if digit in s:
            return float(digit)
    return np.nan


def parse_N_to_num(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip().upper()
    s = s.replace("CLINICAL", "").replace("C", "")

    if "0" in s:
        return 0.0
    if "1" in s:
        return 1.0
    return np.nan


# ============================================================
# r computation
# ============================================================
def compute_r(df: pd.DataFrame, v_cr: float) -> pd.DataFrame:
    out = df.copy()

    event_col = pick_col(out, S.event_col_candidates, required=True)
    time_col = pick_col(out, S.time_col_candidates, required=True)

    is_event = standardize_event_indicator(out[event_col]).to_numpy(dtype=int)
    t = pd.to_numeric(out[time_col], errors="coerce").to_numpy(dtype=float)

    d1_col, d2_col, d3_col = S.prostate_dims
    for col in (d1_col, d2_col, d3_col):
        if col not in out.columns:
            raise KeyError(f"Missing prostate dimension column: {col}")

    d1 = pd.to_numeric(out[d1_col], errors="coerce").to_numpy(dtype=float)
    d2 = pd.to_numeric(out[d2_col], errors="coerce").to_numpy(dtype=float)
    d3 = pd.to_numeric(out[d3_col], errors="coerce").to_numpy(dtype=float)

    v_prostate = (math.pi / 6.0) * d1 * d2 * d3

    if S.suv_primary_preferred in out.columns:
        suv = pd.to_numeric(out[S.suv_primary_preferred], errors="coerce").to_numpy(dtype=float)
        suv_source = S.suv_primary_preferred
    else:
        suv_cols = [c for c in out.columns if str(c).startswith(S.suv_prefix)]
        if not suv_cols:
            raise KeyError("No SUV column found.")
        suv_mat = np.vstack([
            pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)
            for c in suv_cols
        ]).T
        suv = np.nanmax(suv_mat, axis=1)
        suv_source = "max(diagnose_SUVmax.*)"

    v0 = v_prostate * suv
    numerator = np.log1p(v_cr) - np.log1p(v0)

    r_back = np.full(out.shape[0], np.nan, dtype=float)
    r_upper = np.full(out.shape[0], np.nan, dtype=float)

    valid = np.isfinite(t) & (t > 0) & np.isfinite(numerator)
    event_idx = (is_event == 1) & valid
    censor_idx = (is_event == 0) & valid

    r_back[event_idx] = numerator[event_idx] / t[event_idx]
    r_upper[censor_idx] = numerator[censor_idx] / t[censor_idx]

    out["V_prostate"] = v_prostate
    out["SUVmax_primary_used"] = suv
    out["SUVmax_primary_source"] = suv_source
    out["V0"] = v0
    out["r_back"] = r_back
    out["r_upper"] = r_upper
    out["V_CR_used"] = float(v_cr)

    return out


# ============================================================
# Projector
# ============================================================
def extract_covariates(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    age_col = pick_col(df, S.age_col_candidates, required=True)
    isup_col = pick_col(df, S.isup_col_candidates, required=True)
    t_col = pick_col(df, S.t_col_candidates, required=True)
    n_col = pick_col(df, S.n_col_candidates, required=True)

    age = pd.to_numeric(df[age_col], errors="coerce").to_numpy(dtype=float)
    isup = pd.to_numeric(df[isup_col], errors="coerce").to_numpy(dtype=float)
    t_num = df[t_col].apply(parse_T_to_num).to_numpy(dtype=float)
    n_num = df[n_col].apply(parse_N_to_num).to_numpy(dtype=float)

    source_cols = {
        "age": age_col,
        "ISUP": isup_col,
        "T_num": t_col,
        "N_num": n_col,
    }
    return age, isup, t_num, n_num, source_cols


def build_design_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    age, isup, t_num, n_num, source_cols = extract_covariates(df)
    age_std, age_mean, age_sd = zfit(age)

    X = np.column_stack([age_std, isup, t_num, n_num])

    scaler = {
        "feature_order": ["age", "ISUP", "T_num", "N_num"],
        "age_mean": age_mean,
        "age_sd": age_sd,
        "source_cols": source_cols,
    }
    return X, scaler


def build_design_matrix_from_scaler(df: pd.DataFrame, scaler: Dict) -> np.ndarray:
    age, isup, t_num, n_num, _ = extract_covariates(df)

    age_mean = float(scaler["age_mean"])
    age_sd = float(scaler["age_sd"])
    if not np.isfinite(age_sd) or age_sd <= 0:
        age_sd = 1.0

    age_std = (age - age_mean) / age_sd
    X = np.column_stack([age_std, isup, t_num, n_num])
    return X


def neg_loglik(theta: np.ndarray, X: np.ndarray, r_back: np.ndarray, r_upper: np.ndarray, is_event: np.ndarray) -> float:
    intercept = theta[0]
    coef = theta[1:-1]
    log_sigma = theta[-1]
    sigma = math.exp(log_sigma) + S.eps_sigma

    mu = intercept + X @ coef

    event_mask = is_event.astype(bool)
    censor_mask = ~event_mask

    ll = 0.0

    if event_mask.any():
        z_event = (r_back[event_mask] - mu[event_mask]) / sigma
        ll += np.sum(norm.logpdf(z_event) - log_sigma)

    if censor_mask.any():
        z_censor = (r_upper[censor_mask] - mu[censor_mask]) / sigma
        ll += np.sum(norm.logcdf(z_censor))

    return float(-ll)


def fit_mu_projector(df: pd.DataFrame) -> Dict:
    np.random.seed(S.seed)

    event_col = pick_col(df, S.event_col_candidates, required=True)
    is_event = standardize_event_indicator(df[event_col]).to_numpy(dtype=int)

    r_back = pd.to_numeric(df["r_back"], errors="coerce").to_numpy(dtype=float)
    r_upper = pd.to_numeric(df["r_upper"], errors="coerce").to_numpy(dtype=float)

    X, scaler = build_design_matrix(df)

    valid_X = np.isfinite(X).all(axis=1)
    valid_y = ((is_event == 1) & np.isfinite(r_back)) | ((is_event == 0) & np.isfinite(r_upper))
    valid = valid_X & valid_y

    X_fit = X[valid]
    is_event_fit = is_event[valid]
    r_back_fit = r_back[valid]
    r_upper_fit = r_upper[valid]

    p = X_fit.shape[1]
    theta0 = np.zeros(1 + p + 1, dtype=float)

    rb_event = r_back_fit[is_event_fit == 1]
    sigma0 = float(np.nanstd(rb_event, ddof=0)) if rb_event.size > 5 else 0.5
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = 0.5
    theta0[-1] = math.log(sigma0)

    res = minimize(
        fun=neg_loglik,
        x0=theta0,
        args=(X_fit, r_back_fit, r_upper_fit, is_event_fit),
        method="L-BFGS-B",
        options={"maxiter": S.maxiter},
    )

    if not res.success:
        raise RuntimeError(f"Projector optimization failed: {res.message}")

    theta_hat = res.x
    intercept = float(theta_hat[0])
    coef = theta_hat[1:-1].astype(float)
    sigma_hat = float(math.exp(theta_hat[-1]) + S.eps_sigma)

    beta = {
        "Intercept": intercept,
        "age": float(coef[0]),
        "ISUP": float(coef[1]),
        "T_num": float(coef[2]),
        "N_num": float(coef[3]),
    }

    aux = {
        "sigma_hat": sigma_hat,
        "n_fit": int(valid.sum()),
        "n_event_fit": int((is_event_fit == 1).sum()),
        "n_censor_fit": int((is_event_fit == 0).sum()),
        "optimizer": "L-BFGS-B",
        "negloglik": float(res.fun),
        "converged": bool(res.success),
    }

    return {
        "beta": beta,
        "scaler": scaler,
        "aux": aux,
    }


def project_mu(df: pd.DataFrame, beta: Dict, scaler: Dict) -> pd.DataFrame:
    X = build_design_matrix_from_scaler(df, scaler)

    intercept = float(beta["Intercept"])
    coef = np.array(
        [beta["age"], beta["ISUP"], beta["T_num"], beta["N_num"]],
        dtype=float,
    )

    mu = intercept + X @ coef

    out = df.copy()
    out["risk_score_mu"] = mu
    return out


# ============================================================
# Cox
# ============================================================
def fit_cox_on_mu(df: pd.DataFrame, time_col: str, event_col: str, mu_col: str = "risk_score_mu") -> Dict:
    tmp = df.copy()
    tmp["_time"] = pd.to_numeric(tmp[time_col], errors="coerce")
    tmp["_event"] = standardize_event_indicator(tmp[event_col]).astype(int)
    tmp["_mu"] = pd.to_numeric(tmp[mu_col], errors="coerce")

    tmp = tmp.loc[np.isfinite(tmp["_time"]) & np.isfinite(tmp["_mu"])].copy()

    mu_z, mu_mean, mu_sd = zfit(tmp["_mu"].to_numpy(dtype=float))
    tmp["mu_z"] = mu_z

    cph = CoxPHFitter()
    cph.fit(
        tmp[["mu_z", "_time", "_event"]].rename(columns={"_time": "T", "_event": "E"}),
        duration_col="T",
        event_col="E",
        robust=True,
    )

    beta = float(cph.params_["mu_z"])
    hr = float(np.exp(beta))

    ci = cph.confidence_intervals_.loc["mu_z"].to_numpy(dtype=float)
    ci_hr = (float(np.exp(ci[0])), float(np.exp(ci[1])))

    p_wald = float(cph.summary.loc["mu_z", "p"])
    try:
        p_lrt = float(cph.log_likelihood_ratio_test().p_value)
    except Exception:
        p_lrt = np.nan

    return {
        "HR_per_1SD_mu": hr,
        "CI95_lower": ci_hr[0],
        "CI95_upper": ci_hr[1],
        "p_wald": p_wald,
        "p_lrt": p_lrt,
        "mu_mean": mu_mean,
        "mu_sd": mu_sd,
        "n": int(tmp.shape[0]),
        "events": int(tmp["_event"].sum()),
    }


# ============================================================
# Export
# ============================================================
def export_projector(beta: Dict, scaler: Dict, aux: Dict, out_dir: Path) -> None:
    save_tsv(
        pd.DataFrame({"term": list(beta.keys()), "coef": list(beta.values())}),
        out_dir / "projector_beta.tsv",
    )
    save_json(scaler, out_dir / "projector_scaler.json")
    save_json(aux, out_dir / "projector_aux.json")


# ============================================================
# Main
# ============================================================
def main() -> None:
    np.random.seed(S.seed)

    input_path = Path(S.input_path)
    output_dir = Path(S.output_dir)
    ensure_dir(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_raw = try_read_csv(str(input_path))

    event_col = pick_col(df_raw, S.event_col_candidates, required=True)
    time_col = pick_col(df_raw, S.time_col_candidates, required=True)

    df_r = compute_r(df_raw, v_cr=S.v_cr)
    fit = fit_mu_projector(df_r)
    df_out = project_mu(df_r, beta=fit["beta"], scaler=fit["scaler"])
    cox_summary = fit_cox_on_mu(df_out, time_col=time_col, event_col=event_col)

    export_cols = [
        c for c in [
            event_col,
            time_col,
            "V_prostate",
            "SUVmax_primary_used",
            "V0",
            "r_back",
            "r_upper",
            "risk_score_mu",
        ]
        if c in df_out.columns
    ]

    save_tsv(df_out, output_dir / "cohort_with_mu.tsv")
    save_tsv(df_out[export_cols].copy(), output_dir / "cohort_core_results.tsv")
    export_projector(fit["beta"], fit["scaler"], fit["aux"], output_dir)
    save_tsv(pd.DataFrame([cox_summary]), output_dir / "cox_mu_summary.tsv")
    save_json(asdict(S), output_dir / "run_settings.json")

    print("[OK] analysis completed")
    print(f"[OK] input   : {input_path}")
    print(f"[OK] output  : {output_dir}")
    print(f"[OK] n       : {df_out.shape[0]}")
    print(f"[OK] events  : {int(standardize_event_indicator(df_out[event_col]).sum())}")


if __name__ == "__main__":
    main()