# -*- coding: utf-8 -*-

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(stringr)
})

TCGA_DIR   <- "data/TCGA"
COHORT_DIR <- "data/cohort"
OUT_DIR    <- "results/TCGA"

EXPR_PATH        <- file.path(TCGA_DIR, "TCGA_PRAD_counts_tumor_barcode.tsv")
SAMPLE_META_PATH <- file.path(TCGA_DIR, "TCGA_PRAD_samples_tumor_meta.tsv")
CLINICAL_PATH    <- file.path(TCGA_DIR, "clinical.tsv")

PROJ_BETA   <- file.path(COHORT_DIR, "projector_beta.tsv")
PROJ_SCALER <- file.path(COHORT_DIR, "projector_scaler.tsv")
PROJ_AUX    <- file.path(COHORT_DIR, "projector_aux.tsv")

GENESET_PROLIF <- c("AURKA","BIRC5","CDC20","CDK1","FOXM1","MYBL2","PLK1","TOP2A","UBE2C")
GENESET_AR     <- c("AR","KLK3","KLK2","TMPRSS2","FKBP5","NKX3-1","ACSL3","ABCC4","STEAP1")


dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)


safe_fwrite <- function(x, path){
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  fwrite(x, file = path, sep = "\t", quote = FALSE, na = "NA")
}

safe_writeLines <- function(x, path){
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(x, con = path, useBytes = TRUE)
}

make_uid <- function(ids){
  suffix <- ave(seq_along(ids), ids, FUN = function(z) sprintf("%03d", seq_along(z)))
  paste0(ids, "__", suffix)
}

safe_first <- function(x){
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  x[1]
}

to_symbol <- function(x){
  x <- as.character(x)
  ifelse(grepl("\\|", x), sub("^[^|]+\\|", "", x), x)
}

parse_gleason <- function(x){
  x <- as.character(x)
  x[x == ""] <- NA
  x <- gsub(" ", "", x)
  
  out <- rep(NA_real_, length(x))
  
  idx1 <- grepl("^[0-9]+\\+[0-9]+$", x)
  if (any(idx1, na.rm = TRUE)){
    out[idx1] <- sapply(strsplit(x[idx1], "\\+"), function(v){
      if (length(v) == 2 && all(grepl("^[0-9]+$", v))) sum(as.numeric(v)) else NA_real_
    })
  }
  
  idx2 <- !idx1 & grepl("^[0-9]+$", x)
  out[idx2] <- suppressWarnings(as.numeric(x[idx2]))
  out
}

age_to_years <- function(x){
  x <- suppressWarnings(as.numeric(x))
  if (all(is.na(x))) return(x)
  med <- suppressWarnings(median(x, na.rm = TRUE))
  ifelse(is.na(x), NA_real_, ifelse(med > 200, x / 365.25, x))
}

derive_isup <- function(g_group, gleason_num){
  gg <- suppressWarnings(as.numeric(as.character(g_group)))
  out <- gg
  miss <- is.na(out)
  
  if (any(miss)){
    gn <- gleason_num[miss]
    out[miss] <- dplyr::case_when(
      is.na(gn) ~ NA_real_,
      gn <= 6   ~ 1,
      gn == 7   ~ 2,
      gn == 8   ~ 4,
      gn >= 9   ~ 5,
      TRUE ~ NA_real_
    )
  }
  out
}

read_beta <- function(path){
  dt <- fread(path)
  nm <- names(dt)
  
  if (all(c("term", "coef") %in% nm)) {
    return(dt[, .(term, coef)])
  }
  if (all(c("term", "beta_hat") %in% nm)) {
    return(dt[, .(term, coef = beta_hat)])
  }
  
  term_col <- nm[which.max(sapply(nm, function(z) grepl("term|name|feature", z, ignore.case = TRUE)))]
  val_col  <- nm[which.max(sapply(nm, function(z) grepl("coef|beta|estimate|value", z, ignore.case = TRUE)))]
  
  dt[, .(
    term = as.character(get(term_col)),
    coef = suppressWarnings(as.numeric(get(val_col)))
  )]
}

read_kv <- function(path){
  dt <- fread(path)
  nm <- names(dt)
  
  if (all(c("term","value") %in% nm)) {
    return(dt[, .(term, value)])
  }
  
  dt[, .(
    term = as.character(get(nm[1])),
    value = as.character(get(nm[2]))
  )]
}

get_kv_num <- function(kv, key){
  vv <- kv$value[kv$term == key][1]
  suppressWarnings(as.numeric(vv))
}

purity_from_estimate <- function(estimate_score){
  cos(0.6049872018 + 0.0001467884 * estimate_score)
}

write_gct_strict <- function(mat, outfile){
  stopifnot(is.matrix(mat))
  stopifnot(!is.null(rownames(mat)), !is.null(colnames(mat)))
  
  mat <- mat[is.finite(rowMeans(mat)), , drop = FALSE]
  rn <- rownames(mat)
  rn[rn == "" | is.na(rn)] <- NA
  keep <- !is.na(rn)
  mat <- mat[keep, , drop = FALSE]
  rn  <- rn[keep]
  
  lines <- vector("character", length = nrow(mat) + 3L)
  lines[1] <- "#1.2"
  lines[2] <- paste(nrow(mat), ncol(mat), sep = "\t")
  lines[3] <- paste(c("NAME", "Description", colnames(mat)), collapse = "\t")
  
  for (i in seq_len(nrow(mat))){
    vals <- format(mat[i, ], scientific = FALSE, trim = TRUE)
    lines[i + 3] <- paste(c(rn[i], rn[i], vals), collapse = "\t")
  }
  
  safe_writeLines(lines, outfile)
}

read_gct_scores <- function(gct_file){
  x <- readLines(gct_file, warn = FALSE)
  if (length(x) < 4) stop("GCT file too short: ", gct_file)
  
  header <- strsplit(x[3], "\t", fixed = TRUE)[[1]]
  dat_lines <- x[-c(1,2,3)]
  spl <- strsplit(dat_lines, "\t", fixed = TRUE)
  lens <- vapply(spl, length, integer(1))
  
  if (any(lens != length(header))) {
    stop("GCT parse error: inconsistent column counts.")
  }
  
  m <- do.call(rbind, spl)
  colnames(m) <- header
  as.data.frame(m, stringsAsFactors = FALSE, check.names = FALSE)
}

compute_proxy <- function(mat, geneset){
  sym <- to_symbol(rownames(mat))
  keep <- sym %in% geneset
  if (sum(keep) < 3) {
    return(setNames(rep(NA_real_, ncol(mat)), colnames(mat)))
  }
  
  subm <- mat[keep, , drop = FALSE]
  sub_sym <- sym[keep]
  
  if (any(duplicated(sub_sym))){
    dt <- as.data.table(subm)
    dt[, symbol := sub_sym]
    subm2 <- dt[, lapply(.SD, mean, na.rm = TRUE), by = symbol]
    rn <- subm2$symbol
    subm2$symbol <- NULL
    subm <- as.matrix(subm2)
    rownames(subm) <- rn
  }
  
  subm_z <- t(scale(t(subm)))
  sc <- as.numeric(colMeans(subm_z, na.rm = TRUE))
  setNames(sc, colnames(mat))
}

spearman_raw <- function(df, x, y){
  d <- df[, c(x, y), drop = FALSE]
  d <- d[complete.cases(d), , drop = FALSE]
  
  if (nrow(d) < 20) {
    return(list(n = nrow(d), rho = NA_real_, p = NA_real_))
  }
  
  ct <- suppressWarnings(cor.test(d[[x]], d[[y]], method = "spearman", exact = FALSE))
  list(n = nrow(d), rho = unname(ct$estimate), p = ct$p.value)
}

partial_spearman_rankresid <- function(df, x, y, covs){
  d <- df[, c(x, y, covs), drop = FALSE]
  d <- d[complete.cases(d), , drop = FALSE]
  
  if (nrow(d) < 30) {
    return(list(n = nrow(d), rho = NA_real_, p = NA_real_))
  }
  
  rk <- as.data.frame(lapply(d, function(v) rank(v, ties.method = "average")))
  f_cov <- paste(covs, collapse = " + ")
  
  rx <- resid(lm(as.formula(paste0(x, " ~ ", f_cov)), data = rk))
  ry <- resid(lm(as.formula(paste0(y, " ~ ", f_cov)), data = rk))
  
  ct <- suppressWarnings(cor.test(rx, ry, method = "spearman", exact = FALSE))
  list(n = nrow(d), rho = unname(ct$estimate), p = ct$p.value)
}

lm_one <- function(df, formula_str, term = "risk_score_mu"){
  mf <- model.frame(as.formula(formula_str), data = df, na.action = na.omit)
  
  if (nrow(mf) < 30) {
    return(list(n = nrow(mf), beta = NA_real_, se = NA_real_, p = NA_real_, r2 = NA_real_))
  }
  
  fit <- lm(as.formula(formula_str), data = df)
  sm <- summary(fit)
  cf <- coef(sm)
  
  if (!term %in% rownames(cf)) {
    return(list(n = nrow(mf), beta = NA_real_, se = NA_real_, p = NA_real_, r2 = sm$r.squared))
  }
  
  list(
    n    = nrow(mf),
    beta = unname(cf[term, "Estimate"]),
    se   = unname(cf[term, "Std. Error"]),
    p    = unname(cf[term, "Pr(>|t|)"]),
    r2   = unname(sm$r.squared)
  )
}

assoc_one <- function(df, yname){
  sp  <- spearman_raw(df, "risk_score_mu", yname)
  ps  <- partial_spearman_rankresid(df, "risk_score_mu", yname, c("stromal_score", "purity"))
  lm0 <- lm_one(df, paste0(yname, " ~ risk_score_mu"))
  lm1 <- lm_one(df, paste0(yname, " ~ risk_score_mu + stromal_score + purity"))
  
  list(
    spearman_n   = sp$n,
    spearman_rho = sp$rho,
    spearman_p   = sp$p,
    partial_n    = ps$n,
    partial_rho  = ps$rho,
    partial_p    = ps$p,
    lm_raw_n     = lm0$n,
    lm_raw_beta  = lm0$beta,
    lm_raw_se    = lm0$se,
    lm_raw_p     = lm0$p,
    lm_raw_r2    = lm0$r2,
    lm_adj_n     = lm1$n,
    lm_adj_beta  = lm1$beta,
    lm_adj_se    = lm1$se,
    lm_adj_p     = lm1$p,
    lm_adj_r2    = lm1$r2
  )
}


message("[1/6] Reading projector ...")

beta_dt   <- read_beta(PROJ_BETA)
scaler_kv <- read_kv(PROJ_SCALER)
aux_kv    <- tryCatch(read_kv(PROJ_AUX), error = function(e) NULL)

age_mean <- get_kv_num(scaler_kv, "age_mean")
age_sd   <- get_kv_num(scaler_kv, "age_sd")

if (!is.finite(age_mean) || !is.finite(age_sd) || age_sd <= 0){
  stop("projector_scaler.tsv must contain valid age_mean and age_sd")
}

b0   <- beta_dt$coef[beta_dt$term == "Intercept"][1]
bAge <- beta_dt$coef[beta_dt$term == "age"][1]
bI   <- beta_dt$coef[beta_dt$term == "ISUP"][1]
bT   <- beta_dt$coef[beta_dt$term == "T_num"][1]
bN   <- beta_dt$coef[beta_dt$term == "N_num"][1]

if (any(!is.finite(c(b0, bAge, bI, bT, bN)))) {
  stop("projector_beta.tsv is missing Intercept/age/ISUP/T_num/N_num")
}


message("[2/6] Reading expression and sample meta ...")

expr_raw <- fread(EXPR_PATH) |> as.data.frame()
gene_ids <- expr_raw[[1]]
expr_raw[[1]] <- NULL

dup_flag <- duplicated(gene_ids)
if (any(dup_flag)){
  expr_raw <- expr_raw[!dup_flag, , drop = FALSE]
  gene_ids <- gene_ids[!dup_flag]
}

rownames(expr_raw) <- gene_ids
expr <- as.matrix(expr_raw)
storage.mode(expr) <- "numeric"

sm <- fread(SAMPLE_META_PATH) |> as.data.frame()
sample_id_col <- names(sm)[which.max(sapply(sm, function(x) sum(grepl("^TCGA-", as.character(x)))))]

sm$Sample_full <- as.character(sm[[sample_id_col]])
sm$Sample_ID   <- substr(sm$Sample_full, 1, 12)

expr_short <- substr(colnames(expr), 1, 12)
common_ids <- intersect(expr_short, sm$Sample_ID)
if (length(common_ids) == 0) {
  stop("counts and sample_meta cannot be aligned by 12-char TCGA ID")
}

expr <- expr[, match(common_ids, expr_short), drop = FALSE]
sm   <- sm[match(common_ids, sm$Sample_ID), , drop = FALSE]

sm$Sample_UID <- make_uid(sm$Sample_ID)
colnames(expr) <- sm$Sample_UID

stopifnot(ncol(expr) == nrow(sm))
stopifnot(!any(duplicated(colnames(expr))))
stopifnot(all(colnames(expr) == sm$Sample_UID))


message("[3/6] Reading clinical and computing mu ...")

clin <- fread(CLINICAL_PATH) |> as.data.frame()
colnames(clin) <- make.names(colnames(clin), unique = TRUE)

if (!"cases.submitter_id" %in% names(clin)) {
  stop("clinical.tsv must contain cases.submitter_id")
}

clin$Patient_12 <- substr(as.character(clin$cases.submitter_id), 1, 12)

if ("project.project_id" %in% names(clin)) {
  clin <- clin[is.na(clin$project.project_id) | clin$project.project_id == "TCGA-PRAD", , drop = FALSE]
}

need_cols <- c(
  "diagnoses.age_at_diagnosis",
  "diagnoses.gleason_score","diagnoses.primary_gleason_grade","diagnoses.secondary_gleason_grade","diagnoses.gleason_grade_group",
  "diagnoses.ajcc_pathologic_t","diagnoses.ajcc_clinical_t","diagnoses.uicc_pathologic_t","diagnoses.uicc_clinical_t",
  "diagnoses.ajcc_pathologic_n","diagnoses.ajcc_clinical_n","diagnoses.uicc_pathologic_n","diagnoses.uicc_clinical_n"
)
for (cc in need_cols) if (!cc %in% names(clin)) clin[[cc]] <- NA

clin_agg <- clin %>%
  group_by(Patient_12) %>%
  summarise(
    age_dx_raw = safe_first(diagnoses.age_at_diagnosis),
    g_score     = safe_first(diagnoses.gleason_score),
    g_primary   = safe_first(diagnoses.primary_gleason_grade),
    g_secondary = safe_first(diagnoses.secondary_gleason_grade),
    g_group     = safe_first(diagnoses.gleason_grade_group),
    t_path   = safe_first(diagnoses.ajcc_pathologic_t),
    t_clin   = safe_first(diagnoses.ajcc_clinical_t),
    t_uicc_p = safe_first(diagnoses.uicc_pathologic_t),
    t_uicc_c = safe_first(diagnoses.uicc_clinical_t),
    n_path   = safe_first(diagnoses.ajcc_pathologic_n),
    n_clin   = safe_first(diagnoses.ajcc_clinical_n),
    n_uicc_p = safe_first(diagnoses.uicc_pathologic_n),
    n_uicc_c = safe_first(diagnoses.uicc_clinical_n),
    .groups = "drop"
  ) %>%
  mutate(
    age_years = age_to_years(age_dx_raw),
    gleason_raw = dplyr::coalesce(as.character(g_score), as.character(g_primary), as.character(g_group)),
    gleason_num = parse_gleason(gleason_raw),
    T_raw = dplyr::coalesce(as.character(t_path), as.character(t_clin), as.character(t_uicc_p), as.character(t_uicc_c)),
    N_raw = dplyr::coalesce(as.character(n_path), as.character(n_clin), as.character(n_uicc_p), as.character(n_uicc_c)),
    tstage_main = str_extract(T_raw, "T[0-4]"),
    n_stage     = str_extract(N_raw, "N[0-3]")
  )

meta0 <- sm %>%
  left_join(clin_agg, by = c("Sample_ID" = "Patient_12")) %>%
  mutate(
    ISUP = derive_isup(g_group, gleason_num),
    T_num = suppressWarnings(as.numeric(str_extract(tstage_main, "[0-4]"))),
    N_num = case_when(
      n_stage == "N0" ~ 0,
      n_stage %in% c("N1","N2","N3") ~ 1,
      TRUE ~ NA_real_
    ),
    age = age_years,
    age_z = (age - age_mean) / age_sd,
    risk_score_mu = b0 + bAge * age_z + bI * ISUP + bT * T_num + bN * N_num
  )


message("[4/6] Normalizing expression and computing bulk proxies ...")

use_vst <- FALSE
vst_mat <- NULL

if (requireNamespace("DESeq2", quietly = TRUE) &&
    requireNamespace("SummarizedExperiment", quietly = TRUE)) {
  
  suppressPackageStartupMessages(library(DESeq2))
  suppressPackageStartupMessages(library(SummarizedExperiment))
  
  colData <- data.frame(dummy = rep(1, ncol(expr)))
  rownames(colData) <- colnames(expr)
  
  dds <- DESeq2::DESeqDataSetFromMatrix(
    countData = round(expr),
    colData = colData,
    design = ~1
  )
  dds <- DESeq2::estimateSizeFactors(dds)
  vsd <- DESeq2::vst(dds, blind = TRUE)
  vst_mat <- SummarizedExperiment::assay(vsd)
  use_vst <- TRUE
  
} else {
  vst_mat <- log2(expr + 1)
}

prolif_sc <- compute_proxy(vst_mat, GENESET_PROLIF)
ar_sc     <- compute_proxy(vst_mat, GENESET_AR)

meta0$prolif_proxy <- unname(prolif_sc[meta0$Sample_UID])
meta0$ar_proxy     <- unname(ar_sc[meta0$Sample_UID])


message("[5/6] Running ESTIMATE ...")

meta0$stromal_score  <- NA_real_
meta0$immune_score   <- NA_real_
meta0$estimate_score <- NA_real_
meta0$purity         <- NA_real_

estimate_ok <- FALSE
estimate_err <- NA_character_
estimate_backend <- NA_character_

if (requireNamespace("estimate", quietly = TRUE)) {
  suppressPackageStartupMessages(library(estimate))
  estimate_backend <- "estimate"
  
  tryCatch({
    sym <- toupper(to_symbol(rownames(vst_mat)))
    ok <- !is.na(sym) & sym != ""
    mat_sym <- vst_mat[ok, , drop = FALSE]
    rownames(mat_sym) <- sym[ok]
    
    dt <- as.data.table(mat_sym)
    dt[, symbol := rownames(mat_sym)]
    mat_sym2 <- dt[, lapply(.SD, mean, na.rm = TRUE), by = symbol]
    rn <- mat_sym2$symbol
    mat_sym2$symbol <- NULL
    mat_sym2 <- as.matrix(mat_sym2)
    rownames(mat_sym2) <- rn
    
    uid <- colnames(mat_sym2)
    safe_id <- make.names(uid, unique = TRUE)
    map_dt <- data.table(Sample_UID = uid, safe_id = safe_id)
    safe_fwrite(map_dt, file.path(OUT_DIR, "ESTIMATE_sample_id_map.tsv"))
    
    colnames(mat_sym2) <- safe_id
    
    gct_in  <- file.path(OUT_DIR, "ESTIMATE_input.gct")
    gct_flt <- file.path(OUT_DIR, "ESTIMATE_input_common.gct")
    gct_out <- file.path(OUT_DIR, "ESTIMATE_scores.gct")
    
    write_gct_strict(mat_sym2, gct_in)
    estimate::filterCommonGenes(input.f = gct_in, output.f = gct_flt, id = "GeneSymbol")
    estimate::estimateScore(input.ds = gct_flt, output.ds = gct_out, platform = "illumina")
    
    sc_df <- read_gct_scores(gct_out)
    row_id <- sc_df[["NAME"]]
    score_mat <- as.matrix(sc_df[, -(1:2), drop = FALSE])
    rownames(score_mat) <- row_id
    storage.mode(score_mat) <- "numeric"
    
    colnames(score_mat) <- map_dt$Sample_UID[match(colnames(score_mat), map_dt$safe_id)]
    
    if ("StromalScore" %in% rownames(score_mat)) {
      meta0$stromal_score <- unname(score_mat["StromalScore", meta0$Sample_UID])
    }
    if ("ImmuneScore" %in% rownames(score_mat)) {
      meta0$immune_score <- unname(score_mat["ImmuneScore", meta0$Sample_UID])
    }
    if ("ESTIMATEScore" %in% rownames(score_mat)) {
      meta0$estimate_score <- unname(score_mat["ESTIMATEScore", meta0$Sample_UID])
      meta0$purity <- purity_from_estimate(meta0$estimate_score)
    }
    
    estimate_ok <- TRUE
  }, error = function(e){
    estimate_ok <<- FALSE
    estimate_err <<- e$message
  })
}

if (!estimate_ok) {
  estimate_backend <- "tidyestimate"
  
  tryCatch({
    if (!requireNamespace("tidyestimate", quietly = TRUE)) {
      stop("tidyestimate not installed")
    }
    
    suppressPackageStartupMessages(library(tidyestimate))
    
    sym <- toupper(to_symbol(rownames(vst_mat)))
    ok <- !is.na(sym) & sym != ""
    mat_sym <- vst_mat[ok, , drop = FALSE]
    rownames(mat_sym) <- sym[ok]
    
    df_expr <- as.data.frame(mat_sym) %>%
      tibble::rownames_to_column("hgnc_symbol") %>%
      group_by(hgnc_symbol) %>%
      summarise(across(where(is.numeric), mean, na.rm = TRUE), .groups = "drop")
    
    df_flt <- tidyestimate::filter_common_genes(
      df_expr, id = "hgnc_symbol", tidy = TRUE, tell_missing = TRUE, find_alias = FALSE
    )
    sc <- tidyestimate::estimate_score(df_flt, is_affymetrix = FALSE)
    
    sample_col <- intersect(c("sample","tumor","NAME","Sample","samples"), colnames(sc))[1]
    if (is.na(sample_col)) sample_col <- colnames(sc)[1]
    colnames(sc)[colnames(sc) == sample_col] <- "Sample_UID"
    
    meta0 <- meta0 %>% left_join(sc, by = "Sample_UID")
    
    if ("stromal" %in% names(meta0) && all(is.na(meta0$stromal_score))) {
      meta0$stromal_score <- meta0$stromal
    }
    if ("immune" %in% names(meta0) && all(is.na(meta0$immune_score))) {
      meta0$immune_score <- meta0$immune
    }
    if ("estimate" %in% names(meta0) && all(is.na(meta0$estimate_score))) {
      meta0$estimate_score <- meta0$estimate
    }
    
    if (any(is.finite(meta0$estimate_score)) && all(is.na(meta0$purity))) {
      meta0$purity <- purity_from_estimate(meta0$estimate_score)
    }
    
    estimate_ok <- TRUE
    estimate_err <- NA_character_
  }, error = function(e){
    estimate_ok <<- FALSE
    estimate_err <<- paste0("tidyestimate failed: ", e$message)
  })
}


message("[6/6] Running association analysis ...")

df_ok <- meta0 %>% filter(is.finite(risk_score_mu))

a_ar <- assoc_one(df_ok, "ar_proxy")
a_pr <- assoc_one(df_ok, "prolif_proxy")

assoc_tab <- data.table(
  endpoint = c("AR/lineage proxy", "Proliferation proxy"),
  spearman_n   = c(a_ar$spearman_n, a_pr$spearman_n),
  spearman_rho = c(a_ar$spearman_rho, a_pr$spearman_rho),
  spearman_p   = c(a_ar$spearman_p, a_pr$spearman_p),
  partial_n    = c(a_ar$partial_n, a_pr$partial_n),
  partial_rho  = c(a_ar$partial_rho, a_pr$partial_rho),
  partial_p    = c(a_ar$partial_p, a_pr$partial_p),
  lm_raw_n     = c(a_ar$lm_raw_n, a_pr$lm_raw_n),
  lm_raw_beta  = c(a_ar$lm_raw_beta, a_pr$lm_raw_beta),
  lm_raw_se    = c(a_ar$lm_raw_se, a_pr$lm_raw_se),
  lm_raw_p     = c(a_ar$lm_raw_p, a_pr$lm_raw_p),
  lm_raw_r2    = c(a_ar$lm_raw_r2, a_pr$lm_raw_r2),
  lm_adj_n     = c(a_ar$lm_adj_n, a_pr$lm_adj_n),
  lm_adj_beta  = c(a_ar$lm_adj_beta, a_pr$lm_adj_beta),
  lm_adj_se    = c(a_ar$lm_adj_se, a_pr$lm_adj_se),
  lm_adj_p     = c(a_ar$lm_adj_p, a_pr$lm_adj_p),
  lm_adj_r2    = c(a_ar$lm_adj_r2, a_pr$lm_adj_r2)
)

diag_tab <- data.table(
  n_total = nrow(meta0),
  n_mu = sum(is.finite(meta0$risk_score_mu)),
  n_complete_baseline = sum(is.finite(meta0$age) & is.finite(meta0$ISUP) & is.finite(meta0$T_num) & is.finite(meta0$N_num)),
  use_vst = use_vst,
  estimate_ok = estimate_ok,
  estimate_backend = estimate_backend,
  estimate_err = estimate_err,
  stromal_nonNA = sum(is.finite(meta0$stromal_score)),
  purity_nonNA = sum(is.finite(meta0$purity)),
  estimate_nonNA = sum(is.finite(meta0$estimate_score))
)

projector_tab <- data.table(
  term = c("age_mean", "age_sd", "Intercept", "age", "ISUP", "T_num", "N_num"),
  value = c(age_mean, age_sd, b0, bAge, bI, bT, bN)
)

safe_fwrite(as.data.table(meta0), file.path(OUT_DIR, "TCGA_PRAD_meta_full.tsv"))
safe_fwrite(assoc_tab, file.path(OUT_DIR, "Table_TCGA_association.tsv"))
safe_fwrite(diag_tab, file.path(OUT_DIR, "diagnostic_summary.tsv"))
safe_fwrite(projector_tab, file.path(OUT_DIR, "projector_summary.tsv"))

cat("\n[OK] TCGA core analysis finished\n")
cat("Output:", OUT_DIR, "\n")
cat("Samples:", nrow(meta0), "\n")
cat("mu non-missing:", sum(is.finite(meta0$risk_score_mu)), "\n\n")