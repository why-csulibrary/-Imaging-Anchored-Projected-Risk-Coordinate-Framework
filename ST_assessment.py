# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr


KEY_DATA_DIR = "data/ST"
OUT_DIR = "results/ST"

BIN_PX = 256
MIN_SPOTS_PER_BIN = 3

PROGRAMS = ["AR", "NE", "Prolif", "EMT", "Immune", "Fibro", "Endo"]
TME_PROGRAMS = ["Fibro", "Immune", "Endo"]

NEIGHBOR_RADIUS_BINS = 3
BOOTSTRAP_N = 1000
PERM_N = 1000
SEED = 20260128


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv(path):
    return pd.read_csv(path)


def read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(
            f"Cannot read parquet: {path}\n"
            f"Please install pyarrow or fastparquet.\n"
            f"Original error: {e}"
        )


def save_tsv(df, path):
    ensure_dir(Path(path).parent)
    df.to_csv(path, sep="\t", index=False)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pick_first_existing(root, names):
    for name in names:
        fp = Path(root) / name
        if fp.exists():
            return fp
    return None


def clean_barcode(x):
    return str(x).strip()


def load_scalefactors(key_dir):
    fp = pick_first_existing(key_dir, ["scalefactors_json.json", "scalefactors_json"])
    if fp is None:
        raise FileNotFoundError("Missing scalefactors_json.json")
    with open(fp, "r", encoding="utf-8") as f:
        sf = json.load(f)
    return sf, fp


def load_positions(key_dir):
    pos_csv = pick_first_existing(key_dir, ["tissue_positions_list.csv", "tissue_positions.csv"])
    pos_pq = pick_first_existing(key_dir, ["tissue_positions.parquet", "tissue_positions.parq"])

    if pos_csv is None and pos_pq is None:
        raise FileNotFoundError("Missing tissue_positions_list.csv or tissue_positions.parquet")

    if pos_csv is not None:
        tmp = pd.read_csv(pos_csv)
        need = ["barcode", "in_tissue", "pxl_row_in_fullres", "pxl_col_in_fullres"]

        if all(c in tmp.columns for c in need):
            pos = tmp.copy()
            src = pos_csv
        else:
            tmp = pd.read_csv(pos_csv, header=None)
            if tmp.shape[1] != 6:
                raise ValueError(f"Unexpected positions CSV format: {pos_csv}")
            tmp.columns = [
                "barcode", "in_tissue", "array_row", "array_col",
                "pxl_row_in_fullres", "pxl_col_in_fullres"
            ]
            pos = tmp.copy()
            src = pos_csv
    else:
        pos = read_parquet(pos_pq)
        src = pos_pq

    pos["barcode"] = pos["barcode"].map(clean_barcode)
    pos["in_tissue"] = pd.to_numeric(pos["in_tissue"], errors="coerce").fillna(0).astype(int)
    pos["pxl_row_in_fullres"] = pd.to_numeric(pos["pxl_row_in_fullres"], errors="coerce")
    pos["pxl_col_in_fullres"] = pd.to_numeric(pos["pxl_col_in_fullres"], errors="coerce")

    pos = pos.replace([np.inf, -np.inf], np.nan)
    pos = pos.dropna(subset=["pxl_row_in_fullres", "pxl_col_in_fullres"]).copy()

    return pos, str(src)


def pick_base_image_and_scale(key_dir, scalefactors):
    hires_fp = Path(key_dir) / "tissue_hires_image.png"
    lowres_fp = Path(key_dir) / "tissue_lowres_image.png"

    if hires_fp.exists():
        img_fp = hires_fp
        scale_key = "tissue_hires_scalef"
    elif lowres_fp.exists():
        img_fp = lowres_fp
        scale_key = "tissue_lowres_scalef"
    else:
        raise FileNotFoundError("Missing tissue_hires_image.png or tissue_lowres_image.png")

    if scale_key not in scalefactors:
        raise KeyError(f"Missing {scale_key} in scalefactors_json.json")

    scale = float(scalefactors[scale_key])
    img = Image.open(img_fp).convert("RGB")
    return img, str(img_fp), scale_key, scale


def load_scores(key_dir):
    fp = Path(key_dir) / "spot_scores_z.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}")

    sc = read_csv(fp)

    if "barcode" not in sc.columns:
        if "Unnamed: 0" in sc.columns:
            sc = sc.rename(columns={"Unnamed: 0": "barcode"})
        else:
            sc = sc.rename(columns={sc.columns[0]: "barcode"})

    sc["barcode"] = sc["barcode"].map(clean_barcode)

    for prog in PROGRAMS:
        target = f"{prog}_score_z"
        if target in sc.columns:
            continue
        alt = [c for c in sc.columns if str(c).lower() == target.lower()]
        if len(alt) > 0:
            sc = sc.rename(columns={alt[0]: target})

    present = [f"{p}_score_z" for p in PROGRAMS if f"{p}_score_z" in sc.columns]
    if len(present) == 0:
        raise ValueError("No *_score_z columns found in spot_scores_z.csv")

    return sc


def build_spot_table(scores, positions, base_img, scale):
    df = scores.merge(
        positions[["barcode", "in_tissue", "pxl_row_in_fullres", "pxl_col_in_fullres"]],
        on="barcode",
        how="inner"
    )
    if df.shape[0] == 0:
        raise RuntimeError("scores and positions do not overlap on barcode")

    df = df.loc[df["in_tissue"] == 1].copy()

    df["px"] = pd.to_numeric(df["pxl_col_in_fullres"], errors="coerce") * float(scale)
    df["py"] = pd.to_numeric(df["pxl_row_in_fullres"], errors="coerce") * float(scale)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["px", "py"]).copy()

    w, h = base_img.size
    df = df.loc[(df["px"] >= 0) & (df["px"] < w) & (df["py"] >= 0) & (df["py"] < h)].copy()

    return df


def add_bins(df, bin_px):
    out = df.copy()
    out["bx"] = np.floor(out["px"].to_numpy(dtype=float) / float(bin_px)).astype(int)
    out["by"] = np.floor(out["py"].to_numpy(dtype=float) / float(bin_px)).astype(int)
    return out


def aggregate_bin_means(df, score_cols, bin_px, min_spots=3):
    g_mean = df.groupby(["bx", "by"], as_index=False)[score_cols].mean()
    g_n = df.groupby(["bx", "by"], as_index=False).size().rename(columns={"size": "n_spots"})

    out = g_mean.merge(g_n, on=["bx", "by"], how="left")
    out = out.loc[out["n_spots"] >= int(min_spots)].copy()
    out["x"] = (out["bx"].astype(float) + 0.5) * float(bin_px)
    out["y"] = (out["by"].astype(float) + 0.5) * float(bin_px)
    return out


def program_correlation(bin_df, score_cols):
    rows = []
    for c1 in score_cols:
        for c2 in score_cols:
            tmp = bin_df[[c1, c2]].copy().dropna()
            if tmp.shape[0] < 3:
                rho = np.nan
                p = np.nan
                n = int(tmp.shape[0])
            else:
                rho, p = spearmanr(tmp[c1], tmp[c2], nan_policy="omit")
                n = int(tmp.shape[0])
            rows.append({
                "program_x": c1.replace("_score_z", ""),
                "program_y": c2.replace("_score_z", ""),
                "n": n,
                "rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            })
    return pd.DataFrame(rows)


def neighbor_pairs(bin_df, radius_bins):
    coords = bin_df[["bx", "by"]].to_numpy(dtype=int)
    key_to_idx = {(int(bx), int(by)): i for i, (bx, by) in enumerate(coords)}

    neigh = []
    for bx, by in coords:
        ids = []
        for dx in range(-radius_bins, radius_bins + 1):
            for dy in range(-radius_bins, radius_bins + 1):
                if dx == 0 and dy == 0:
                    continue
                j = key_to_idx.get((int(bx + dx), int(by + dy)))
                if j is not None:
                    ids.append(j)
        neigh.append(np.array(ids, dtype=int))
    return neigh


def compute_neighbor_other_indices(bin_df, hot_idx, radius_bins):
    n = bin_df.shape[0]
    neigh_list = neighbor_pairs(bin_df, radius_bins)

    hot_set = set(map(int, np.asarray(hot_idx, dtype=int).tolist()))
    nb_set = set()
    for i in hot_set:
        for j in neigh_list[i]:
            nb_set.add(int(j))
    nb_set -= hot_set

    all_set = set(range(n))
    other_set = all_set - hot_set - nb_set

    nb_idx = np.array(sorted(nb_set), dtype=int)
    other_idx = np.array(sorted(other_set), dtype=int)
    return nb_idx, other_idx


def bootstrap_mean_diff(x1, x0, n_boot=1000, seed=1):
    rng = np.random.default_rng(seed)
    x1 = np.asarray(x1, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    x1 = x1[np.isfinite(x1)]
    x0 = x0[np.isfinite(x0)]

    if x1.size < 3 or x0.size < 3:
        return np.nan, np.nan, np.nan

    obs = float(np.mean(x1) - np.mean(x0))
    boots = np.empty(int(n_boot), dtype=float)

    for b in range(int(n_boot)):
        s1 = rng.choice(x1, size=x1.size, replace=True)
        s0 = rng.choice(x0, size=x0.size, replace=True)
        boots[b] = float(np.mean(s1) - np.mean(s0))

    lo, hi = np.quantile(boots, [0.025, 0.975])
    return obs, float(lo), float(hi)


def tme_neighbor_enrichment(bin_df, target_prog="Prolif", thr_q=0.92, radius_bins=3, n_boot=1000, seed=1):
    tcol = f"{target_prog}_score_z"
    if tcol not in bin_df.columns:
        raise ValueError(f"Missing {tcol}")

    v = bin_df[tcol].to_numpy(dtype=float)
    thr = float(np.nanquantile(v, thr_q))
    hot_idx = np.where(v >= thr)[0]

    if hot_idx.size < 5:
        thr = float(np.nanquantile(v, 0.985))
        hot_idx = np.where(v >= thr)[0]
    if hot_idx.size < 5:
        raise ValueError("Too few hotspot bins")

    nb_idx, other_idx = compute_neighbor_other_indices(bin_df, hot_idx, radius_bins)

    rows = []
    for prog in TME_PROGRAMS:
        col = f"{prog}_score_z"
        if col not in bin_df.columns:
            continue

        x1 = bin_df.iloc[nb_idx][col].to_numpy(dtype=float)
        x0 = bin_df.iloc[other_idx][col].to_numpy(dtype=float)

        est, lo, hi = bootstrap_mean_diff(
            x1, x0,
            n_boot=n_boot,
            seed=seed + (abs(hash(prog)) % 100000)
        )
        rows.append({
            "program": prog,
            "mean_diff_neighbor_minus_other": est,
            "ci_low": lo,
            "ci_high": hi,
            "threshold": thr,
            "n_hot": int(hot_idx.size),
            "n_neighbor": int(nb_idx.size),
            "n_other": int(other_idx.size),
        })

    return pd.DataFrame(rows)


def moran_I(values, coords_bxby):
    v = np.asarray(values, dtype=float)
    coords = np.asarray(coords_bxby, dtype=int)

    ok = np.isfinite(v)
    v = v[ok]
    coords = coords[ok]

    n = v.size
    if n < 10:
        return np.nan

    z = v - np.mean(v)
    key_to_i = {(int(bx), int(by)): i for i, (bx, by) in enumerate(coords)}

    W = 0.0
    num = 0.0
    for i, (bx, by) in enumerate(coords):
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            j = key_to_i.get((int(bx + dx), int(by + dy)))
            if j is None:
                continue
            W += 1.0
            num += z[i] * z[j]

    den = np.sum(z ** 2)
    if den <= 0 or W <= 0:
        return np.nan

    return float((n / W) * (num / den))


def moran_permutation(values, coords_bxby, n_perm=1000, seed=1):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    coords = np.asarray(coords_bxby, dtype=int)

    obs = moran_I(v, coords)
    null = np.empty(int(n_perm), dtype=float)

    for b in range(int(n_perm)):
        perm = rng.permutation(v)
        null[b] = moran_I(perm, coords)

    null = null[np.isfinite(null)]
    if np.isfinite(obs) and null.size > 0:
        p = (np.sum(null >= obs) + 1) / (null.size + 1)
    else:
        p = np.nan

    return float(obs) if np.isfinite(obs) else np.nan, float(p) if np.isfinite(p) else np.nan


def moran_summary(bin_df):
    coords = bin_df[["bx", "by"]].to_numpy(dtype=int)
    rows = []

    for prog in ["Prolif", "AR"]:
        col = f"{prog}_score_z"
        if col not in bin_df.columns:
            rows.append({
                "program": prog,
                "observed_I": np.nan,
                "p_value_perm": np.nan,
                "n_bins": int(bin_df.shape[0]),
            })
            continue

        obs, p = moran_permutation(
            bin_df[col].to_numpy(dtype=float),
            coords,
            n_perm=PERM_N,
            seed=SEED
        )
        rows.append({
            "program": prog,
            "observed_I": obs,
            "p_value_perm": p,
            "n_bins": int(bin_df.shape[0]),
        })

    return pd.DataFrame(rows)


def main():
    ensure_dir(OUT_DIR)

    key_dir = Path(KEY_DATA_DIR)

    scalefactors, scalefactor_file = load_scalefactors(key_dir)
    positions, positions_source = load_positions(key_dir)
    scores = load_scores(key_dir)

    base_img, image_file, scale_key, scale_val = pick_base_image_and_scale(key_dir, scalefactors)
    spots = build_spot_table(scores, positions, base_img, scale_val)

    score_cols = [f"{p}_score_z" for p in PROGRAMS if f"{p}_score_z" in spots.columns]
    spots_b = add_bins(spots, BIN_PX)
    bin_df = aggregate_bin_means(spots_b, score_cols, bin_px=BIN_PX, min_spots=MIN_SPOTS_PER_BIN)

    corr_df = program_correlation(bin_df, score_cols)
    enrich_df = tme_neighbor_enrichment(
        bin_df,
        target_prog="Prolif",
        thr_q=0.92,
        radius_bins=NEIGHBOR_RADIUS_BINS,
        n_boot=BOOTSTRAP_N,
        seed=SEED,
    )
    moran_df = moran_summary(bin_df)

    save_tsv(spots, f"{OUT_DIR}/spots_used.tsv")
    save_tsv(bin_df, f"{OUT_DIR}/bin_means.tsv")
    save_tsv(corr_df, f"{OUT_DIR}/program_correlation.tsv")
    save_tsv(enrich_df, f"{OUT_DIR}/tme_neighbor_enrichment.tsv")
    save_tsv(moran_df, f"{OUT_DIR}/moran_summary.tsv")

    run_info = {
        "key_data_dir": str(key_dir),
        "scalefactors_file": scalefactor_file,
        "positions_source": positions_source,
        "base_image": image_file,
        "base_image_width": int(base_img.size[0]),
        "base_image_height": int(base_img.size[1]),
        "scale_key_used": scale_key,
        "scale_value_used": float(scale_val),
        "bin_px": int(BIN_PX),
        "min_spots_per_bin": int(MIN_SPOTS_PER_BIN),
        "n_spots_used": int(spots.shape[0]),
        "n_bins_used": int(bin_df.shape[0]),
        "programs_present": [c.replace("_score_z", "") for c in score_cols],
    }
    save_json(run_info, f"{OUT_DIR}/run_info.json")

    print("[OK] ST core analysis finished")
    print(f"[OK] spots used: {spots.shape[0]}")
    print(f"[OK] bins used : {bin_df.shape[0]}")
    print(f"[OK] output    : {OUT_DIR}")


if __name__ == "__main__":
    main()