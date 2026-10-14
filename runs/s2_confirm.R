# Study 2 confirmatory analysis (drafts/study2-spec.md section 9). To be frozen with stamp A.
#
#   Rscript runs/s2_confirm.R out_dir table.csv [table2.csv ...] [options]
#
# Tables come from runs/s2_table.py --phase confirm (one per model or run; pilot tables are refused, the pilot is
# never pooled with confirmatory data). Options, each filled only from rule outputs at stamp B:
#   --dose qwen=2,k26=4      confirmatory dose per model (rule 1; default the registered dose)
#   --drop qwen:N03,k26:joy  vectors dropped at the gates (rule 1); every test and permutation is recomputed
#   --dropped k26            a model dropped from S2-H1 before any confirmatory result was read: fewer than 7
#                            surviving denial arguments, or formally abandoned in DEVIATIONS-s2.md (the other model
#                            gets 0.05)
#   --k26 holm|abandoned     holm (default); abandoned is the same as --dropped k26
#   --descriptive k26        a model that left the Holm pair under rule 1 (CI only; the other model gets 0.05)
# S2-H1 alpha is standard Holm over the pair (both models unless dropped or descriptive), whatever the order the
# models were computed in: the smaller p is tested at 0.025 and, if it passes, the other at 0.05. A pair model with
# no S2-H1 result (not in the tables, or no fit) counts as p = 1, so the other is tested at 0.025.
# Gating: each follow-up test opens only if the test before it was significant with a positive estimate (S2-G1
# after S2-H1, S2-G2 after S2-G1, S2-G8 after S2-G2, S2-G7 after S2-G6). A closed test is reported as not tested
# (significant and equivalent NA).
#   --kr                     Kenward-Roger df and p for the two S2-H1 fits, reported beside Satterthwaite
#   --only-h1                fit S2-H1 only (used to time the Kenward-Roger check on a full-size synthetic table)
#   --no-loo                 skip the leave-one-vector-out refits (descriptive)
#   --potency potency.csv    per-vector potency (columns model, vec, potency; optional dose), the change in
#                            next-token NLL on neutral text under the vector (spec section 9 moderators), as
#                            `runs/s2_run.py potency` writes it; without the file the potency rows are skipped.
# Writes, in out_dir: hypotheses.csv (every registered test: S2-H1, S2-R1, S2-G1, S2-G2, S2-G8, S2-G4, S2-G5,
# S2-G6, S2-G7, S2-H8 per model, with the alpha used, permutation p, governing p and reading), headline.csv
# (decision and interpretation tables), descriptive.csv (CI-only estimates), vectors.csv, linearity.csv,
# mass.csv (Yes+No mass and mass drop by subject and vector class, capacity and persona items), flips.csv
# (argmax flip rates by vector class, exploratory), openings.csv (per opening: reply length, emotion-word count,
# subject mentions in the wording, opening-level X), fallbacks.csv (every fit's fallback step and warnings, for
# DEVIATIONS-s2.md) and kr.csv with --kr. Nothing written to descriptive.csv, mass.csv, flips.csv or openings.csv
# feeds a registered test.
# Needs R >= 4.3 with lme4, lmerTest (and pbkrtest for --kr).

# Single-threaded BLAS: multithreaded OpenBLAS made one S2-H1 fit about 20x slower on a 10-core Mac (38 s vs
# 1.7 s at 3,360 rows). OpenBLAS reads the setting at load, so the script re-executes itself once with it set.
if (!interactive() && Sys.getenv("OPENBLAS_NUM_THREADS") != "1") {
  Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")
  self <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
  quit(save = "no", status = system2(file.path(R.home("bin"), "Rscript"), shQuote(c(self, commandArgs(TRUE)))))
}
suppressMessages(library(lmerTest))

# ---- registered constants -----------------------------------------------------------------------------
EMO <- list(qwen = c("joy", "sadness", "anger", "fear", "euphoria", "surprise"),
            k26 = c("joy", "fear", "euphoria", "surprise"))
LAYER <- c(qwen = 40, k26 = 20)
DOSE <- c(qwen = 2, k26 = 4)
PRIMARY <- c("noun", "arithmetic", "noise")  # the S2-H1 control class
# valence moderator (exploratory): surprise is in neither group; each model uses the emotions of EMO it has
VALENCE <- list(positive = c("joy", "euphoria", "calm"), negative = c("sadness", "anger", "fear"))
SESOI <- 0.10
ALPHA <- 0.05  # familywise, two-sided
DEPTH_MAX <- 15  # interpretation table: both |s0| below 15 in S2-DEN
MASS_MIN <- 0.9
# Study 1 self-minus-others, emotions minus (red, ocean), under prompted denial (spec section 16, item 9)
STUDY1 <- list(qwen = c(raw = 0.12, raw_lo = 0.09, raw_hi = 0.15, corr = -0.01, corr_lo = -0.03, corr_hi = 0.02),
               k26 = c(raw = 0.14, raw_lo = 0.11, raw_hi = 0.16, corr = 0.11, corr_lo = 0.09, corr_hi = 0.14))

# ---- options ------------------------------------------------------------------------------------------
opt <- list(dose = DOSE, drop = list(), k26 = "holm", dropped = character(0), descriptive = character(0), kr = FALSE,
            only_h1 = FALSE, loo = TRUE, potency = NULL)
parse_args <- function(a) {
  files <- character(0); i <- 1
  while (i <= length(a)) {
    k <- a[i]
    if (k == "--dose") { for (kv in strsplit(a[i + 1], ",")[[1]]) { p <- strsplit(kv, "=")[[1]]; opt$dose[p[1]] <<- as.numeric(p[2]) }; i <- i + 2 }
    else if (k == "--drop") { for (kv in strsplit(a[i + 1], ",")[[1]]) { p <- strsplit(kv, ":")[[1]]; opt$drop[[p[1]]] <<- c(opt$drop[[p[1]]], paste(p[-1], collapse = ":")) }; i <- i + 2 }
    else if (k == "--k26") {
      if (a[i + 1] == "qwen-first") stop("--k26 qwen-first was removed: S2-H1 uses standard Holm over the pair whatever the order of computation")
      opt$k26 <<- match.arg(a[i + 1], c("holm", "abandoned")); if (opt$k26 == "abandoned") opt$dropped <<- union(opt$dropped, "k26"); i <- i + 2
    }
    else if (k == "--dropped") { opt$dropped <<- union(opt$dropped, strsplit(a[i + 1], ",")[[1]]); i <- i + 2 }
    else if (k == "--descriptive") { opt$descriptive <<- c(opt$descriptive, strsplit(a[i + 1], ",")[[1]]); i <- i + 2 }
    else if (k == "--kr") { opt$kr <<- TRUE; i <- i + 1 }
    else if (k == "--only-h1") { opt$only_h1 <<- TRUE; i <- i + 1 }
    else if (k == "--no-loo") { opt$loo <<- FALSE; i <- i + 1 }
    else if (k == "--potency") { opt$potency <<- a[i + 1]; i <- i + 2 }
    else { files <- c(files, k); i <- i + 1 }
  }
  files
}

# ---- data ---------------------------------------------------------------------------------------------
tab <- NULL
FALLBACKS <- list()
VERBOSE <- TRUE

sel <- function(model, arm, measure, classes, vecs = NULL) {
  t <- tab[tab$model == model & tab$arm == arm & tab$measure == measure & tab$vclass %in% classes &
           tab$layer == LAYER[[model]] & abs(tab$dose - opt$dose[[model]]) < 1e-9 &
           !(tab$vec %in% opt$drop[[model]]), ]
  if (!is.null(vecs)) t <- t[t$vec %in% vecs, ]
  t
}

# cells of one measure: one row per (vector, argument, wording, paraphrase), twin-averaged by s2_table.py
cells <- function(model, arm, measure, classes, vecs = NULL) {
  t <- sel(model, arm, measure, classes, vecs)
  if (!nrow(t)) return(NULL)
  emo <- intersect(EMO[[model]], unique(t$vec[t$vclass == "emotion"]))
  d <- data.frame(vector = factor(t$vec), vclass = t$vclass, argument = factor(t$argument),
                  aw = factor(paste(t$argument, t$wording, sep = "_w")), para = factor(t$para),
                  opening = t$opening, arm = t$arm, gen = t$generator, kind = t$arg_kind, y = t$value,
                  x = as.numeric(t$vclass == "emotion"), stringsAsFactors = FALSE)
  for (e in emo) d[[paste0("E_", e)]] <- as.numeric(t$vec == e)
  d$ctl <- 1 - d$x
  attr(d, "emo") <- emo
  d
}

# attach a per-cell measure (flag, mass, gap0 ...) to cells of another measure
with_cell <- function(d, model, arm, measure, name) {
  t <- sel(model, arm, measure, unique(d$vclass))
  key <- function(v, o, p) paste(v, o, p, sep = "|")
  d[[name]] <- t$value[match(key(d$vector, d$opening, d$para), key(t$vec, t$opening, t$para))]
  d
}

# ---- model fitting with the registered fallback order -------------------------------------------------
# chain(): the S2-H1 random part and its fallbacks, cumulative: drop correlations; drop the argument:wording
# slope; drop the paraphrase slope; drop vector:paraphrase. The control-vector term and vector:argument are
# never dropped. vt / vt_nc: the vector term with and without correlations; arg / arg_nc: the argument terms.
chain <- function(vt = "(0 + ctl | vector)", vt_nc = vt, arg = "(1 + x | argument)", arg_nc = "(1 + x || argument)") {
  base <- c(vt, "(1 | vector:argument)", "(1 | vector:para)", arg, "(1 + x | aw)", "(1 + x | para)", "(1 | aw:para)")
  nc <- c(vt_nc, "(1 | vector:argument)", "(1 | vector:para)", arg_nc, "(1 + x || aw)", "(1 + x || para)", "(1 | aw:para)")
  s3 <- replace(nc, 5, "(1 | aw)")
  s4 <- replace(s3, 6, "(1 | para)")
  s5 <- s4[-3]
  list(maximal = base, no_corr = nc, no_aw_slope = s3, no_para_slope = s4, no_vector_para = s5)
}

fit_chain <- function(d, fixed, ch, label) {
  last <- tail(names(ch), 1)
  for (step in names(ch)) {
    warn <- NULL
    f <- as.formula(paste("y ~", fixed, "+", paste(ch[[step]], collapse = " + ")))
    m <- tryCatch(withCallingHandlers(
      suppressMessages(lmer(f, data = d, REML = TRUE, control = lmerControl(check.conv.singular = "ignore"))),
      warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") }),
      error = function(e) { warn <<- c(warn, paste("error:", conditionMessage(e))); NULL })
    ok <- !is.null(m) && is.null(warn)
    if (VERBOSE) message(sprintf("%s: %s %s", label, step, if (ok) "ok" else "failed"))
    FALLBACKS[[length(FALLBACKS) + 1]] <<- data.frame(test = label, step = step, converged = ok,
                                                      warnings = paste(unique(warn), collapse = " | "))
    if (ok || (step == last && !is.null(m))) return(list(m = m, step = step, converged = ok))
  }
  NULL
}

# a linear contrast of the fixed effects with Satterthwaite df capped at `cap`
contrast <- function(m, w, cap, ddf = "Satterthwaite") {
  L <- setNames(rep(0, length(fixef(m))), names(fixef(m)))
  L[names(w)] <- w
  ct <- contest(m, L, joint = FALSE, ddf = ddf)
  est <- ct[["Estimate"]]; se <- ct[["Std. Error"]]; df_raw <- ct[["df"]]
  df <- min(df_raw, cap)
  tv <- est / se
  q95 <- qt(0.975, df); q90 <- qt(0.95, df)
  data.frame(estimate = est, se = se, df_satt = df_raw, df = df, t = tv, p_model = 2 * pt(-abs(tv), df),
             ci95_lo = est - q95 * se, ci95_hi = est + q95 * se, ci90_lo = est - q90 * se, ci90_hi = est + q90 * se,
             p_tost = max(pt((est - SESOI) / se, df), pt((est + SESOI) / se, df, lower.tail = FALSE)))
}

# Welch t of vector-level values, emotion vs the other class, over every relabeling (two-sided)
perm_test <- function(v, isE) {
  n <- length(v); k <- sum(isE); nc <- n - k
  if (k < 2 || nc < 2) return(c(p_perm = NA, n_perm = 0))
  welch <- function(se, sse) {
    me <- se / k; mc <- (sum(v) - se) / nc
    ve <- (sse - k * me^2) / (k - 1); vc <- (sum(v^2) - sse - nc * mc^2) / (nc - 1)
    (me - mc) / sqrt(ve / k + vc / nc)
  }
  t0 <- welch(sum(v[isE]), sum(v[isE]^2))
  cmb <- combn(n, k)
  tt <- numeric(ncol(cmb))
  for (j in split(seq_len(ncol(cmb)), ceiling(seq_len(ncol(cmb)) / 2e5))) {
    vv <- matrix(v[cmb[, j]], nrow = k)
    tt[j] <- welch(colSums(vv), colSums(vv^2))
  }
  c(p_perm = mean(abs(tt) >= abs(t0) - 1e-12), n_perm = ncol(cmb))
}

na_row <- function(why) data.frame(estimate = NA, se = NA, df_satt = NA, df = NA, t = NA, p_model = NA, ci95_lo = NA,
                                   ci95_hi = NA, ci90_lo = NA, ci90_hi = NA, p_tost = NA, p_perm = NA, n_perm = 0,
                                   fit = why, converged = NA, singular = NA, n_emo = 0, n_ctl = 0, n_arg = 0,
                                   n_aw = 0, n_para = 0, n_rows = 0)

# The S2-H1 model on cells d: emotions fixed, the other class random. Tested contrast mean_j(E_j) - ctl.
fit_h1 <- function(d, label, perm = TRUE, ddf_kr = FALSE) {
  if (is.null(d) || !nrow(d) || sum(d$x) == 0 || sum(d$ctl) == 0) return(na_row("no data"))
  emo <- attr(d, "emo"); nC <- length(unique(d$vector[d$x == 0]))
  fixed <- paste("0 +", paste0("E_", emo, collapse = " + "), "+ ctl")
  r <- fit_chain(d, fixed, chain(), label)
  if (is.null(r)) return(na_row("no fit"))
  w <- c(setNames(rep(1 / length(emo), length(emo)), paste0("E_", emo)), ctl = -1)
  out <- contrast(r$m, w, nC - 1)
  vm <- tapply(d$y, droplevels(d$vector), mean); isE <- tapply(d$x, droplevels(d$vector), `[`, 1) == 1
  pp <- if (perm) perm_test(as.numeric(vm), as.logical(isE)) else c(p_perm = NA, n_perm = 0)
  res <- cbind(out, p_perm = pp[["p_perm"]], n_perm = pp[["n_perm"]], fit = r$step, converged = r$converged,
               singular = isSingular(r$m), n_emo = length(emo), n_ctl = nC, n_arg = nlevels(droplevels(d$argument)),
               n_aw = nlevels(droplevels(d$aw)), n_para = nlevels(droplevels(d$para)), n_rows = nrow(d))
  if (ddf_kr) {
    t0 <- Sys.time()
    k <- contrast(r$m, w, nC - 1, ddf = "Kenward-Roger")
    attr(res, "kr") <- data.frame(test = label, kr_df = k$df_satt, kr_p_uncapped = 2 * pt(-abs(k$t), k$df_satt),
                                  kr_p_capped = k$p_model, satt_df = out$df_satt, satt_p_capped = out$p_model,
                                  kr_seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")),
                                  n_rows = nrow(d))
  }
  res
}

# S2-G2: d on capacity items (q = 1) and on the non-conceded placebo predicates (q = 0)
fit_g2 <- function(model, arm, label) {
  a <- cells(model, arm, "d", c("emotion", PRIMARY)); b <- cells(model, arm, "d_plc", c("emotion", PRIMARY))
  if (is.null(a) || is.null(b)) return(na_row("no data"))
  emo <- attr(a, "emo")
  a$q <- 1; b$q <- 0
  b$para <- paste0("plc_", b$para)
  d <- rbind(a, b[, names(a)]); d$para <- factor(d$para)
  for (e in emo) d[[paste0("Eq_", e)]] <- d[[paste0("E_", e)]] * d$q
  d$ctlq <- d$ctl * d$q
  nC <- length(unique(d$vector[d$x == 0]))
  fixed <- paste("0 +", paste0("E_", emo, collapse = " + "), "+", paste0("Eq_", emo, collapse = " + "), "+ ctl + ctlq")
  r <- fit_chain(d, fixed, chain("(0 + ctl + ctlq | vector)", "(0 + ctl + ctlq || vector)"), label)
  if (is.null(r)) return(na_row("no fit"))
  out <- contrast(r$m, c(setNames(rep(1 / length(emo), length(emo)), paste0("Eq_", emo)), ctlq = -1), nC - 1)
  vq <- tapply(d$y * ifelse(d$q == 1, 1, -1) / ifelse(d$q == 1, ave(d$q, d$vector, FUN = sum), ave(1 - d$q, d$vector, FUN = sum)),
               droplevels(d$vector), sum)  # per vector: mean d(capacity) - mean d(placebo)
  isE <- tapply(d$x, droplevels(d$vector), `[`, 1) == 1
  pp <- perm_test(as.numeric(vq), as.logical(isE))
  cbind(out, p_perm = pp[["p_perm"]], n_perm = pp[["n_perm"]], fit = r$step, converged = r$converged,
        singular = isSingular(r$m), n_emo = length(emo), n_ctl = nC, n_arg = nlevels(droplevels(d$argument)),
        n_aw = nlevels(droplevels(d$aw)), n_para = nlevels(d$para), n_rows = nrow(d))
}

# X-by-arm (S2-G5: S2-DEN vs S2-CON; S2-G7: S2-FTD vs S2-FTC). Positive = X larger in arm1.
fit_by_arm <- function(model, arm1, arm2, label, covariates = TRUE) {
  a <- cells(model, arm1, "d", c("emotion", PRIMARY)); b <- cells(model, arm2, "d", c("emotion", PRIMARY))
  if (is.null(a) || is.null(b)) return(na_row("no data"))
  a <- with_cell(a, model, arm1, "gap0", "gap"); b <- with_cell(b, model, arm2, "gap0", "gap")
  emo <- intersect(attr(a, "emo"), attr(b, "emo"))
  d <- rbind(a[, intersect(names(a), names(b))], b[, intersect(names(a), names(b))])
  d$A1 <- as.numeric(d$arm == arm1)
  d$gap_c <- d$gap - ave(d$gap, d$arm)
  cov <- "gap_c"
  if (covariates) {
    rl <- tab$reply_len[match(d$opening, tab$opening)]; ew <- tab$emo_words[match(d$opening, tab$opening)]
    if (all(!is.na(rl)) && sd(rl) > 0) { d$reply_len <- as.numeric(scale(rl)); cov <- c(cov, "reply_len") }
    if (all(!is.na(ew)) && sd(ew) > 0) { d$emo_words <- as.numeric(scale(ew)); cov <- c(cov, "emo_words") }
  }
  res <- fit_arm_cells(d, emo, cov, label)
  res$covariates <- paste(cov, collapse = "+")
  res
}

# the X-by-arm model on cells d with an arm indicator A1 (per-emotion fixed means by arm, control-vector slopes
# by arm, argument intercept and x slope by arm, the rest as in S2-H1, covariates as fixed slopes)
fit_arm_cells <- function(d, emo, cov, label, perm = TRUE) {
  d$A2 <- 1 - d$A1
  d$argument <- factor(paste(d$A1, d$argument)); d$aw <- factor(paste(d$A1, d$aw))
  for (e in emo) { d[[paste0("E1_", e)]] <- d[[paste0("E_", e)]] * d$A1; d[[paste0("E2_", e)]] <- d[[paste0("E_", e)]] * d$A2 }
  d$c1 <- d$ctl * d$A1; d$c2 <- d$ctl * d$A2; d$A1x <- d$A1 * d$x; d$A2x <- d$A2 * d$x
  nC <- length(unique(d$vector[d$x == 0]))
  fixed <- paste("0 +", paste0("E1_", emo, collapse = " + "), "+", paste0("E2_", emo, collapse = " + "), "+ c1 + c2",
                 if (length(cov)) paste("+", paste(cov, collapse = " + ")) else "")
  ch <- chain("(0 + c1 + c2 | vector)", "(0 + c1 + c2 || vector)",
              "(0 + A1 + A1x | argument) + (0 + A2 + A2x | argument)", "(0 + A1 + A1x || argument) + (0 + A2 + A2x || argument)")
  r <- fit_chain(d, fixed, ch, label)
  if (is.null(r)) return(na_row("no fit"))
  w <- c(setNames(rep(1 / length(emo), length(emo)), paste0("E1_", emo)), c1 = -1,
         setNames(rep(-1 / length(emo), length(emo)), paste0("E2_", emo)), c2 = 1)
  out <- contrast(r$m, w, nC - 1)
  pp <- c(p_perm = NA, n_perm = 0)
  if (perm) {
    v1 <- tapply(d$y[d$A1 == 1], droplevels(d$vector[d$A1 == 1]), mean); v2 <- tapply(d$y[d$A2 == 1], droplevels(d$vector[d$A2 == 1]), mean)
    k <- intersect(names(v1), names(v2))
    pp <- perm_test(as.numeric(v1[k] - v2[k]), k %in% emo)
  }
  cbind(out, p_perm = pp[["p_perm"]], n_perm = pp[["n_perm"]], fit = r$step, converged = r$converged,
        singular = isSingular(r$m), n_emo = length(emo), n_ctl = nC, n_arg = nlevels(droplevels(d$argument)),
        n_aw = nlevels(droplevels(d$aw)), n_para = nlevels(droplevels(d$para)), n_rows = nrow(d))
}

# t-interval over openings of (mean over emotions - mean over the given controls) of a measure
opening_t <- function(model, arm, measure, ctl_vecs) {
  t <- sel(model, arm, measure, c("emotion", "bridge", PRIMARY))
  t <- t[t$vclass == "emotion" | t$vec %in% ctl_vecs, ]
  if (!nrow(t)) return(c(estimate = NA, ci95_lo = NA, ci95_hi = NA, n = 0))
  pv <- aggregate(value ~ opening + vec + vclass, t, mean)
  e <- tapply(pv$value[pv$vclass == "emotion"], pv$opening[pv$vclass == "emotion"], mean)
  c <- tapply(pv$value[pv$vclass != "emotion"], pv$opening[pv$vclass != "emotion"], mean)
  k <- intersect(names(e), names(c)); z <- e[k] - c[k]
  if (length(z) < 2) return(c(estimate = NA, ci95_lo = NA, ci95_hi = NA, n = length(z)))
  tt <- t.test(z)
  c(estimate = unname(tt$estimate), ci95_lo = tt$conf.int[1], ci95_hi = tt$conf.int[2], n = length(z))
}

# per opening: mean over emotions minus mean over the controls of a measure (each vector averaged over the
# opening's cells first); named by opening
ec_open <- function(model, arm, measure, ctl = PRIMARY) {
  t <- sel(model, arm, measure, c("emotion", ctl))
  if (!nrow(t)) return(NULL)
  pv <- aggregate(value ~ opening + vec + vclass, t, mean)
  e <- tapply(pv$value[pv$vclass == "emotion"], pv$opening[pv$vclass == "emotion"], mean)
  c <- tapply(pv$value[pv$vclass != "emotion"], pv$opening[pv$vclass != "emotion"], mean)
  k <- intersect(names(e), names(c))
  e[k] - c[k]
}

t_over <- function(z, what = "t over openings") {  # one-sample t-interval as a descriptive row
  z <- z[!is.na(z)]
  if (length(z) < 2 || sd(z) == 0) return(data.frame(estimate = if (length(z)) mean(z) else NA, ci95_lo = NA, ci95_hi = NA,
                                                     df = length(z) - 1, fit = what, n_rows = length(z)))
  tt <- t.test(z)
  data.frame(estimate = unname(tt$estimate), ci95_lo = tt$conf.int[1], ci95_hi = tt$conf.int[2], df = length(z) - 1,
             fit = what, n_rows = length(z))
}

ols_slope <- function(y, x, what) {  # slope of y on x with its 95% CI (descriptive)
  ok <- !is.na(y) & !is.na(x)
  if (sum(ok) < 3 || sd(x[ok]) == 0) return(data.frame(estimate = NA, ci95_lo = NA, ci95_hi = NA, df = NA, fit = what, n_rows = sum(ok)))
  f <- lm(y[ok] ~ x[ok]); ci <- confint(f)
  data.frame(estimate = coef(f)[[2]], ci95_lo = ci[2, 1], ci95_hi = ci[2, 2], df = sum(ok) - 2, fit = what, n_rows = sum(ok))
}

# ---- registered tests per model -----------------------------------------------------------------------
row_of <- function(id, family, model, arm, what, predicted, r, governs_perm) {
  r$p <- if (governs_perm) pmax(r$p_model, r$p_perm, na.rm = FALSE) else r$p_model
  cbind(data.frame(id = id, family = family, model = model, arm = arm, what = what, predicted = predicted,
                   perm_governs = governs_perm), r[, setdiff(names(r), "covariates"), drop = FALSE],
        covariates = if ("covariates" %in% names(r)) r$covariates else "")
}

reading <- function(sig, est) {
  if (is.na(sig)) return("not tested")
  if (!sig) return("null")
  if (est < 0) return("significant negative")
  if (est >= SESOI) "supported" else "detectable but below SESOI"
}

# a follow-up test opens only after the test before it was significant with a positive estimate
opens_next <- function(sig, est) isTRUE(sig) && isTRUE(est > 0)

# S2-H1 alphas: standard Holm over the pair (both models minus dropped and descriptive ones). p is named by model;
# a pair model without a result counts as p = 1. Returns alphas named by model (NA outside the pair).
h1_alphas <- function(p, dropped = character(0), descriptive = character(0)) {
  pair <- setdiff(c("qwen", "k26"), c(dropped, descriptive))
  a <- setNames(rep(NA_real_, length(p)), names(p))
  if (!length(pair)) return(a)
  pp <- setNames(rep(1, length(pair)), pair)
  got <- intersect(names(p)[!is.na(p)], pair); pp[got] <- p[got]
  h <- holm_alpha(pp, pair)
  a[intersect(names(p), pair)] <- h[intersect(names(p), pair)]
  a
}

# Holm over the models whose step ran: a step with no data did not run; a step that ran without a fit counts as p = 1
holm_ran <- function(p, fit, names_) holm_alpha(ifelse(is.na(p) & fit != "no data", 1, p), names_)

holm_alpha <- function(p, names_) {  # step-down Holm alphas for named p values at familywise ALPHA
  a <- setNames(rep(NA_real_, length(p)), names_)
  ok <- !is.na(p); if (!any(ok)) return(a)
  o <- order(p[ok]); nm <- names_[ok][o]; m <- sum(ok)
  for (i in seq_along(nm)) {
    a[nm[i]] <- ALPHA / (m - i + 1)
    if (p[ok][o][i] >= a[nm[i]]) { if (i < m) a[nm[(i + 1):m]] <- ALPHA / (m - i + 1); break }
  }
  a
}

main <- function(args) {
  out <- args[1]
  files <- parse_args(args[-1])
  tab <<- do.call(rbind, lapply(files, read.csv, stringsAsFactors = FALSE))
  stopifnot(all(tab$phase == "confirm"), all(tab$layer[tab$model %in% names(LAYER)] == LAYER[tab$model]))
  tab$para <<- as.character(tab$para)
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  models <- intersect(c("qwen", "k26"), unique(tab$model))
  H <- list(); KR <- list(); D <- list()
  add <- function(...) H[[length(H) + 1]] <<- row_of(...)
  desc <- function(model, what, r) {  # one CI-only row with fixed columns
    r <- as.data.frame(r)
    for (c in c("estimate", "ci95_lo", "ci95_hi", "df", "fit", "n_rows")) if (!c %in% names(r)) r[[c]] <- NA
    D[[length(D) + 1]] <<- cbind(data.frame(model = model, what = what), r[, c("estimate", "ci95_lo", "ci95_hi", "df", "fit", "n_rows")])
  }

  # S2-H1
  for (m in models) {
    r <- fit_h1(cells(m, "S2-DEN", "d", c("emotion", PRIMARY)), paste0("S2-H1.", m), ddf_kr = opt$kr)
    if (!is.null(attr(r, "kr"))) KR[[length(KR) + 1]] <- cbind(model = m, attr(r, "kr"))
    add("S2-H1", "primary", m, "S2-DEN", "d, registered emotions - primary controls", "+", r, TRUE)
  }
  H1 <- do.call(rbind, H)
  # alpha for S2-H1: standard Holm over the pair (see --dropped and --descriptive)
  H1$alpha <- unname(h1_alphas(setNames(H1$p, H1$model), opt$dropped, opt$descriptive)[H1$model])
  H1$significant <- ifelse(is.na(H1$alpha), NA, H1$p < H1$alpha)
  H1$reading <- mapply(reading, H1$significant, H1$estimate)
  H1$reading[H1$model %in% opt$descriptive] <- "descriptive (rule 1)"
  H1$reading[H1$model %in% opt$dropped] <- "dropped before any result was read (not tested)"
  if (opt$only_h1) {
    write.csv(H1, file.path(out, "hypotheses.csv"), row.names = FALSE)
    if (length(KR)) write.csv(do.call(rbind, KR), file.path(out, "kr.csv"), row.names = FALSE)
    write.csv(do.call(rbind, FALLBACKS), file.path(out, "fallbacks.csv"), row.names = FALSE)
    print(format(H1[, c("id", "model", "estimate", "ci95_lo", "ci95_hi", "df_satt", "df", "p_model", "p_perm", "p", "alpha", "reading", "fit")], digits = 3), row.names = FALSE)
    if (length(KR)) print(do.call(rbind, KR), digits = 4)
    return(invisible())
  }
  H <- list()
  sig_of <- function(r, alpha) !is.na(r$p) && r$p < alpha
  pos_sig <- function(r, alpha) opens_next(sig_of(r, alpha), r$estimate)
  for (m in models) {
    h1 <- H1[H1$model == m, ]
    h1_sig <- isTRUE(h1$significant) && !(m %in% opt$descriptive)
    h1_open <- h1_sig && opens_next(h1$significant, h1$estimate)
    # S2-R1 depth check (CI only) and the per-argument regression of X on the unsteered gap
    r <- fit_h1(cells(m, "S2-DEN", "d_e", c("emotion", PRIMARY)), paste0("S2-R1.", m), perm = FALSE)
    add("S2-R1", "robustness (CI only)", m, "S2-DEN", "d_e (emotion-item response-curve fit), H1 model", "+", r, FALSE)
    dd <- with_cell(cells(m, "S2-DEN", "d", c("emotion", PRIMARY)), m, "S2-DEN", "gap0", "gap")
    if (!is.null(dd)) {
      ax <- sapply(split(dd, droplevels(dd$argument)), function(z) c(X = mean(z$y[z$x == 1]) - mean(z$y[z$x == 0]), gap = mean(z$gap)))
      if (ncol(ax) >= 3) {
        lmf <- lm(X ~ gap, data.frame(t(ax))); ci <- confint(lmf)
        desc(m, "S2-R1 per-argument: intercept of X on unsteered gap", data.frame(estimate = coef(lmf)[[1]], ci95_lo = ci[1, 1], ci95_hi = ci[1, 2], df = ncol(ax) - 2, fit = "OLS over arguments", n_rows = ncol(ax)))
        desc(m, "S2-R1 per-argument: slope of X on unsteered gap", data.frame(estimate = coef(lmf)[[2]], ci95_lo = ci[2, 1], ci95_hi = ci[2, 2], df = ncol(ax) - 2, fit = "OLS over arguments", n_rows = ncol(ax)))
      }
    }
    # Family 1, fixed order, each gated on the previous step (significant with a positive estimate, S2-H1 included)
    g1 <- fit_h1(cells(m, "S2-DEN", "d", c("emotion", "noun")), paste0("S2-G1.", m))
    add("S2-G1", "family 1", m, "S2-DEN", "d, emotions - sampled nouns only", "+", g1, TRUE)
    g1r <- row_of("", "", m, "", "", "", g1, TRUE)
    g2 <- fit_g2(m, "S2-DEN", paste0("S2-G2.", m))
    add("S2-G2", "family 1", m, "S2-DEN", "X by q: capacity d minus placebo d", "+", g2, FALSE)
    g2r <- row_of("", "", m, "", "", "", g2, FALSE)
    mp <- cells(m, "S2-DEN", "d", c("emotion", "thinking"))
    n_mp <- if (is.null(mp)) 0 else length(unique(mp$vector[mp$x == 0]))
    g8 <- fit_h1(mp, paste0("S2-G8.", m))
    add("S2-G8", if (n_mp >= 5) "family 1" else "family 1 (descriptive, n_mp < 5)", m, "S2-DEN",
        "d, emotions - mental-process controls", "+", g8, n_mp >= 5)
    g8r <- row_of("", "", m, "", "", "", g8, n_mp >= 5)
    k <- length(H)
    H[[k - 2]]$gate_open <- h1_open
    H[[k - 1]]$gate_open <- h1_open && pos_sig(g1r, ALPHA)
    H[[k]]$gate_open <- h1_open && pos_sig(g1r, ALPHA) && pos_sig(g2r, ALPHA)
    H[[k - 2]]$alpha <- H[[k - 1]]$alpha <- H[[k]]$alpha <- ALPHA
    if (n_mp < 5) H[[k]]$alpha <- NA  # descriptive: "significant" reads as CI above 0
    # Family 2 S2-G4 (Holm across models below)
    g4 <- fit_h1(cells(m, "S2-CON", "d", c("emotion", PRIMARY)), paste0("S2-G4.", m))
    add("S2-G4", "family 2", m, "S2-CON", "d, emotions - primary controls, no denial", "+", g4, TRUE)
    # S2-H8 equivalence on both comparators (applies after a non-significant S2-H1)
    for (ms in c("d", "d_other")) {
      e <- fit_h1(cells(m, "S2-DEN", ms, c("emotion", PRIMARY)), paste0("S2-H8.", ms, ".", m), perm = FALSE)
      add("S2-H8", "equivalence", m, "S2-DEN", paste("TOST +-0.10 on", ms), "0", e, FALSE)
      H[[length(H)]]$gate_open <- !h1_sig && !(m %in% opt$dropped)
      H[[length(H)]]$alpha <- ALPHA
    }
    # Qwen fine-tune sequence: S2-G6 at 0.05 (not gated); S2-G7 at 0.05, opens only if S2-G6 is significant
    # with a positive estimate. S2-G7's model is S2-G5's by-arm model with the centred unsteered gap as the only
    # covariate (the two arms share openings and replies, so reply length and emotion-word count are identical).
    if (m == "qwen" && any(tab$arm == "S2-FTD")) {
      g6 <- fit_h1(cells(m, "S2-FTD", "d", c("emotion", PRIMARY)), "S2-G6.qwen")
      add("S2-G6", "fine-tune", m, "S2-FTD", "d, emotions - primary controls, D2 adapter", "+", g6, TRUE)
      H[[length(H)]]$alpha <- ALPHA; H[[length(H)]]$gate_open <- TRUE
      g6r <- row_of("", "", m, "", "", "", g6, TRUE)
      g7 <- fit_by_arm(m, "S2-FTD", "S2-FTC", "S2-G7.qwen", covariates = FALSE)
      add("S2-G7", "fine-tune", m, "S2-FTD vs S2-FTC", "X(FTD) - X(FTC)", "+/-", g7, FALSE)
      H[[length(H)]]$alpha <- ALPHA; H[[length(H)]]$gate_open <- pos_sig(g6r, ALPHA)
      # interpretation precondition: S2-FTD unsteered gap inside S2-DEN's range and covered by the reference set
      og <- function(arm) { t <- sel(m, arm, "gap0", c("emotion", PRIMARY)); tapply(t$value, t$opening, mean) }
      gd <- og("S2-DEN"); gf <- og("S2-FTD")
      lo <- sel(m, "S2-FTD", "ref_lo", c("emotion", PRIMARY)); s0 <- sel(m, "S2-FTD", "s0_you", c("emotion", PRIMARY))
      s1 <- sel(m, "S2-FTD", "s0_lm", c("emotion", PRIMARY))
      covd <- all(tapply(lo$value, lo$opening, min) <= pmin(tapply(s0$value, s0$opening, min), tapply(s1$value, s1$opening, min))[names(tapply(lo$value, lo$opening, min))] + 3)
      pre <- length(gf) && length(gd) && mean(gf) >= min(gd) && mean(gf) <= max(gd) && covd
      for (j in (length(H) - 1):length(H)) H[[j]]$precondition <- pre
    }
  }
  # Family 2 alphas: Holm across models for S2-G4; S2-G5 only where S2-G4 or S2-H1 is significant
  H <- do.call(rbind, lapply(H, function(z) { for (c in c("gate_open", "alpha", "precondition")) if (!c %in% names(z)) z[[c]] <- NA; z }))
  H$gate_open[H$id == "S2-G4"] <- TRUE  # not gated
  i4 <- which(H$id == "S2-G4")
  a4 <- holm_ran(H$p[i4], H$fit[i4], H$model[i4]); H$alpha[i4] <- a4[H$model[i4]]
  G5 <- list()
  for (m in models) {
    g4s <- isTRUE(H$p[H$id == "S2-G4" & H$model == m] < H$alpha[H$id == "S2-G4" & H$model == m])
    h1s <- isTRUE(H1$significant[H1$model == m])
    if (g4s || h1s) {
      g5 <- fit_by_arm(m, "S2-DEN", "S2-CON", paste0("S2-G5.", m))
      G5[[length(G5) + 1]] <- row_of("S2-G5", "family 2", m, "S2-DEN vs S2-CON",
                                     "X(DEN) - X(CON), confounded by emotion vocabulary in the openings", "+/-", g5, FALSE)
    }
  }
  if (length(G5)) {
    G5 <- do.call(rbind, G5); G5$gate_open <- TRUE; G5$precondition <- NA
    a5 <- holm_ran(G5$p, G5$fit, G5$model); G5$alpha <- a5[G5$model]
    H <- rbind(H, G5[, names(H)])
  }
  H1$gate_open <- TRUE; H1$precondition <- NA
  H$reading <- NA
  H <- rbind(H1[, names(H)], H)
  H$significant <- ifelse(is.na(H$alpha), NA, H$p < H$alpha)
  H$significant[H$id == "S2-H8"] <- NA
  H$equivalent <- ifelse(H$id == "S2-H8", H$p_tost < 0.05, NA)
  closed <- !is.na(H$gate_open) & !H$gate_open  # a test that does not open is reported as not tested
  H$significant[closed] <- NA; H$equivalent[closed] <- NA
  H$ci_above_0 <- H$ci95_lo > 0
  write.csv(H, file.path(out, "hypotheses.csv"), row.names = FALSE)

  # ---- decision and interpretation tables (stamped; headline per model) -------------------------------
  hd <- list()
  for (m in models) {
    g <- function(id, col = "significant") { v <- H[[col]][H$id == id & H$model == m]; if (length(v)) v[1] else NA }
    rd <- H1$reading[H1$model == m]
    s0 <- c(mean(sel(m, "S2-DEN", "s0_you", c("emotion", PRIMARY))$value), mean(sel(m, "S2-DEN", "s0_lm", c("emotion", PRIMARY))$value))
    g8sig <- if (is.na(g("S2-G8", "alpha"))) isTRUE(g("S2-G8", "ci_above_0")) else isTRUE(g("S2-G8")) && g("S2-G8", "estimate") > 0
    eq_both <- isTRUE(H$equivalent[H$id == "S2-H8" & H$model == m & grepl("on d$", H$what)]) &&
               isTRUE(H$equivalent[H$id == "S2-H8" & H$model == m & grepl("d_other", H$what)])
    g4pos <- isTRUE(g("S2-G4")) && g("S2-G4", "estimate") > 0
    null_reading <- if (g4pos && eq_both && all(abs(s0) < DEPTH_MAX)) "denial removes self-specific coupling" else "uninformative"
    head <- switch(rd,
      "supported" = {
        if (!isTRUE(g("S2-G1")) || g("S2-G1", "estimate") <= 0) "emotion directions beat a control pool that includes arithmetic and random, but are not distinguishable from randomly sampled neutral-noun controls"
        else if (!isTRUE(g("S2-G2")) || g("S2-G2", "estimate") <= 0) "not distinguishable from a generic subject shift on non-emotion predicates"
        else if (!g8sig) "claim sentence with 'emotion directions' replaced by 'inner-state directions'"
        else "the claim sentence"
      },
      "detectable but below SESOI" = "detectable but below the smallest effect of interest",
      "significant negative" = "emotion activation strengthens the conceded self-denial",
      "null" = paste("null:", null_reading),
      rd)
    if (rd == "supported" && !isTRUE(g("S2-R1", "ci_above_0"))) head <- paste(head, "+ not distinguishable from a depth artifact")
    hd[[length(hd) + 1]] <- data.frame(model = m, s2_h1 = rd, headline = head, s0_you = s0[1], s0_lm = s0[2],
                                       h8_equivalent_both = eq_both, s2_g4_positive = g4pos)
  }
  hd <- do.call(rbind, hd)
  if (nrow(hd) == 2 && sum(hd$s2_h1 == "supported") == 1) hd$combined <- "one model supported, the other not: each reported separately; no general claim about language models"
  else if (nrow(hd) == 1 || length(opt$dropped)) hd$combined <- paste0("one model only (dropped before results: ", if (length(opt$dropped)) paste(opt$dropped, collapse = ", ") else "none declared; the missing model counts as not significant", ")")
  else hd$combined <- ""
  write.csv(hd, file.path(out, "headline.csv"), row.names = FALSE)

  # ---- descriptive tier (CI only) --------------------------------------------------------------------
  ci_of <- function(r) r[, c("estimate", "ci95_lo", "ci95_hi", "df", "fit", "n_rows")]
  for (m in models) {
    P <- c("emotion", PRIMARY)
    for (ms in c("d_raw", "d_state", "d_state_raw", "d_other", "d_you_name", "d_name_lm", "r_rocks", "r_humans",
                 "r_you", "r_lm", "d_pos", "d_neg", "d_prob")) {
      dd <- cells(m, "S2-DEN", ms, P)
      if (!is.null(dd)) desc(m, paste("S2-DEN X on", ms), ci_of(fit_h1(dd, paste0("desc.", ms, ".", m), perm = FALSE)))
    }
    # compression bias of the raw contrast: B = mean(a_E - a_C) x gap
    fa <- sel(m, "S2-DEN", "fit_a", P); gp <- sel(m, "S2-DEN", "gap0", P)
    if (nrow(fa)) desc(m, "raw-contrast compression bias B = (a_E - a_C) x gap",
                       data.frame(estimate = (mean(fa$value[fa$vclass == "emotion"]) - mean(fa$value[fa$vclass != "emotion"])) * mean(gp$value),
                                  ci95_lo = NA, ci95_hi = NA, df = NA, fit = "", n_rows = nrow(fa)))
    # sensitivity: drop flagged cells; restrict to mass >= 0.9
    dd <- with_cell(with_cell(cells(m, "S2-DEN", "d", P), m, "S2-DEN", "flag", "flag"), m, "S2-DEN", "mass", "mass")
    if (!is.null(dd)) {
      keep <- function(z, i) { a <- attr(z, "emo"); z <- z[i, ]; attr(z, "emo") <- a; z }
      desc(m, sprintf("S2-H1 without y0-flagged cells (%d flagged)", sum(dd$flag == 1, na.rm = TRUE)),
           ci_of(fit_h1(keep(dd, dd$flag != 1), paste0("desc.noflag.", m), perm = FALSE)))
      desc(m, "S2-H1 on cells with Yes+No mass >= 0.9", ci_of(fit_h1(keep(dd, dd$mass >= MASS_MIN), paste0("desc.mass.", m), perm = FALSE)))
      for (k in unique(dd$kind)) desc(m, paste("S2-H1 on", k, "arguments"), ci_of(fit_h1(keep(dd, dd$kind == k), paste0("desc.kind.", m), perm = FALSE)))
      for (k in unique(dd$gen)) desc(m, paste("S2-H1 on", k, "wordings"), ci_of(fit_h1(keep(dd, dd$gen == k), paste0("desc.gen.", m), perm = FALSE)))
      if (opt$loo) for (v in levels(droplevels(dd$vector))) {
        z <- keep(dd, dd$vector != v); attr(z, "emo") <- setdiff(attr(dd, "emo"), v)
        if (v %in% attr(dd, "emo")) z[[paste0("E_", v)]] <- NULL
        desc(m, paste("S2-H1 leaving out", v), ci_of(fit_h1(z, paste0("desc.loo.", m), perm = FALSE)))
      }
    }
    # frame: x:frame on paraphrase-averaged cells (state minus capacity), reduced model without paraphrase terms
    a <- sel(m, "S2-DEN", "d", P); b <- sel(m, "S2-DEN", "d_state", P)
    if (nrow(a) && nrow(b)) {
      a <- aggregate(value ~ vec + vclass + argument + wording + opening, a, mean); b <- aggregate(value ~ vec + opening, b, mean)
      z <- merge(a, b, by = c("vec", "opening"), suffixes = c(".c", ".s"))
      d <- data.frame(vector = factor(z$vec), argument = factor(z$argument), aw = factor(paste(z$argument, z$wording)),
                      y = z$value.s - z$value.c, x = as.numeric(z$vclass == "emotion"))
      emo <- intersect(EMO[[m]], unique(z$vec)); for (e in emo) d[[paste0("E_", e)]] <- as.numeric(z$vec == e); d$ctl <- 1 - d$x
      r <- fit_chain(d, paste("0 +", paste0("E_", emo, collapse = " + "), "+ ctl"),
                     list(maximal = c("(0 + ctl | vector)", "(1 + x | argument)", "(1 + x | aw)"),
                          no_corr = c("(0 + ctl | vector)", "(1 + x || argument)", "(1 + x || aw)"),
                          no_aw_slope = c("(0 + ctl | vector)", "(1 + x || argument)", "(1 | aw)")), paste0("desc.frame.", m))
      if (!is.null(r)) desc(m, "x:frame, state minus capacity", cbind(ci_of(cbind(contrast(r$m, c(setNames(rep(1 / length(emo), length(emo)), paste0("E_", emo)), ctl = -1),
                                                                                       length(unique(z$vec[z$vclass != "emotion"])) - 1), fit = r$step, n_rows = nrow(d)))))
    }
    # Study 1 bridge (S2-DEN only): emotions minus red and ocean, t-interval over openings like Study 1's over tasks
    for (ms in c("d_raw", "d")) {
      b <- opening_t(m, "S2-DEN", ms, c("red", "ocean"))
      s1 <- STUDY1[[m]]; s1 <- if (ms == "d") s1[c("corr", "corr_lo", "corr_hi")] else s1[c("raw", "raw_lo", "raw_hi")]
      desc(m, paste("Study 1 bridge, emotions - (red, ocean),", if (ms == "d") "compression-corrected" else "raw"),
           data.frame(estimate = b[["estimate"]], ci95_lo = b[["ci95_lo"]], ci95_hi = b[["ci95_hi"]], df = b[["n"]] - 1,
                      fit = sprintf("Study 1: %+.2f [%+.2f, %+.2f]", s1[1], s1[2], s1[3]), n_rows = b[["n"]]))
    }
    # copying route: per-twin X by closing-ask overlap tercile (content-word share, see runs/s2_table.py), on d
    # and on the language-models rows alone
    for (tw in c("pos", "neg")) for (ms in paste0(c("d_", "r_lm_"), tw)) {
      dd <- cells(m, "S2-DEN", ms, P)
      if (is.null(dd)) next
      dd <- with_cell(dd, m, "S2-DEN", paste0("ovl_", tw), "ovl")
      if (all(is.na(dd$ovl))) next
      br <- unique(quantile(dd$ovl, c(0, 1 / 3, 2 / 3, 1), na.rm = TRUE))
      bin <- if (length(br) > 1) cut(dd$ovl, br, include.lowest = TRUE) else factor(rep(format(br), nrow(dd)))
      for (b in levels(droplevels(bin))) {
        a <- attr(dd, "emo"); z <- dd[!is.na(bin) & bin == b, ]; attr(z, "emo") <- a
        desc(m, sprintf("copying route: X on %s, overlap bin %s", ms, b), ci_of(fit_h1(z, paste0("desc.copy.", m), perm = FALSE)))
      }
    }
    # persona: you-specific shift per vector class
    pt <- sel(m, "S2-DEN", "d_persona", c("emotion", PRIMARY, "thinking", "bridge"))
    for (cl in unique(pt$vclass)) {
      z <- tapply(pt$value[pt$vclass == cl], pt$opening[pt$vclass == cl], mean)
      if (length(z) >= 2) { tt <- t.test(z); desc(m, paste("persona d, class", cl), data.frame(estimate = unname(tt$estimate), ci95_lo = tt$conf.int[1], ci95_hi = tt$conf.int[2], df = length(z) - 1, fit = "t over openings", n_rows = length(z))) }
    }
    # P(Yes) is d_prob above. Argmax flips (exploratory): emotions minus primary controls on directional flips,
    # you minus language models, t over openings
    desc(m, "argmax flips toward emotion present, you - language models, emotions - primary controls", t_over(ec_open(m, "S2-DEN", "d_flip")))
    # persona: Yes+No mass drop (unsteered minus steered) by subject, emotions minus primary controls, on the
    # capacity items and on the persona item; and its subject difference you - language models (predicted ~0)
    for (pre in c("mdrop_", "persona_mdrop_")) {
      frame <- if (pre == "mdrop_") "capacity items" else "persona item"
      sub <- unique(sub(pre, "", tab$measure[startsWith(tab$measure, pre)], fixed = TRUE))
      ez <- lapply(setNames(sub, sub), function(s) ec_open(m, "S2-DEN", paste0(pre, s)))
      for (s in sub) if (!is.null(ez[[s]])) desc(m, sprintf("mass drop, emotions - primary controls, %s, subject %s", frame, s), t_over(ez[[s]]))
      if (!is.null(ez$you) && !is.null(ez$lm)) {
        k <- intersect(names(ez$you), names(ez$lm))
        desc(m, sprintf("mass drop, emotions - primary controls, %s, you - language models", frame), t_over(ez$you[k] - ez$lm[k]))
      }
    }
    # moderators: subject mentions per wording, slope of opening-level X (d) over S2-DEN openings (wordings are
    # nested in arguments; the CI ignores that)
    xo <- ec_open(m, "S2-DEN", "d")
    for (mn in c("wording_lm_mentions", "wording_you_mentions")) {
      w <- sel(m, "S2-DEN", mn, c("emotion", PRIMARY))
      if (nrow(w) && length(xo)) {
        cnt <- tapply(w$value, w$opening, `[`, 1)[names(xo)]
        desc(m, paste("moderator: slope of opening X on", mn), ols_slope(as.numeric(xo), as.numeric(cnt), "OLS over openings"))
      }
    }
    # potency (only with --potency): emotions - primary controls, and per-vector d on potency
    if (!is.null(opt$potency)) {
      po <- read.csv(opt$potency, stringsAsFactors = FALSE)
      po <- po[po$model == m, ]
      if ("dose" %in% names(po)) po <- po[abs(po$dose - opt$dose[[m]]) < 1e-9, ]
      t <- sel(m, "S2-DEN", "d", c("emotion", PRIMARY))
      if (nrow(po) && nrow(t)) {
        vd <- aggregate(value ~ vec + vclass, aggregate(value ~ vec + vclass + opening, t, mean), mean)
        vd$potency <- po$potency[match(vd$vec, po$vec)]
        miss <- vd$vec[is.na(vd$potency)]
        if (length(miss)) warning(sprintf("%s: no potency for %s", m, paste(miss, collapse = ", ")))
        vd <- vd[!is.na(vd$potency), ]
        e <- vd$potency[vd$vclass == "emotion"]; c <- vd$potency[vd$vclass != "emotion"]
        if (length(e) >= 2 && length(c) >= 2) {
          tt <- t.test(e, c)
          desc(m, "potency, emotions - primary controls", data.frame(estimate = mean(e) - mean(c), ci95_lo = tt$conf.int[1], ci95_hi = tt$conf.int[2],
                                                                     df = unname(tt$parameter), fit = "Welch t over vectors", n_rows = nrow(vd)))
        }
        desc(m, "potency: slope of per-vector d on potency, emotions and primary controls", ols_slope(vd$value, vd$potency, "OLS over vectors"))
        if (length(e) >= 2 && length(c) >= 2) {
          f <- lm(value ~ I(vclass == "emotion") + potency, vd); ci <- confint(f)
          desc(m, "potency: X adjusted for potency (per-vector d ~ emotion + potency)",
               data.frame(estimate = coef(f)[[2]], ci95_lo = ci[2, 1], ci95_hi = ci[2, 2], df = f$df.residual, fit = "OLS over vectors", n_rows = nrow(vd)))
        }
      }
    }
    # valence (exploratory): the S2-H1 model refit on the same cells; per emotion E_e - ctl (X for that emotion
    # alone), and mean(E positive) - mean(E negative) over the emotions this model uses; df capped as in S2-H1
    dd <- cells(m, "S2-DEN", "d", P)
    if (!is.null(dd) && sum(dd$x) > 0 && sum(dd$ctl) > 0) {
      emo <- attr(dd, "emo"); nC <- length(unique(dd$vector[dd$x == 0]))
      r <- fit_chain(dd, paste("0 +", paste0("E_", emo, collapse = " + "), "+ ctl"), chain(), paste0("desc.valence.", m))
      vrow <- function(w) ci_of(cbind(contrast(r$m, w, nC - 1), fit = r$step, n_rows = nrow(dd)))
      if (!is.null(r)) {
        for (e in emo) desc(m, sprintf("valence: X for %s alone (%s - primary controls)", e, e), vrow(c(setNames(1, paste0("E_", e)), ctl = -1)))
        pos <- intersect(VALENCE$positive, emo); neg <- intersect(VALENCE$negative, emo)
        if (length(pos) && length(neg))
          desc(m, sprintf("valence: positive (%s) - negative (%s) emotions", paste(pos, collapse = ", "), paste(neg, collapse = ", ")),
               vrow(c(setNames(rep(1 / length(pos), length(pos)), paste0("E_", pos)), setNames(rep(-1 / length(neg), length(neg)), paste0("E_", neg)))))
      }
    }
  }
  if (length(D)) write.csv(do.call(rbind, D), file.path(out, "descriptive.csv"), row.names = FALSE)

  # ---- per opening: reply length, emotion-word count, wording mentions, opening-level X ------------------
  O <- list()
  for (m in models) for (arm in unique(tab$arm[tab$model == m])) {
    t <- tab[tab$model == m & tab$arm == arm & tab$measure == "d" & tab$vclass %in% c("emotion", PRIMARY), ]
    if (!nrow(t)) next
    o <- t[!duplicated(t$opening), c("model", "arm", "argument", "arg_kind", "wording", "generator", "opening", "reply_len", "emo_words")]
    for (mn in c("wording_lm_mentions", "wording_you_mentions")) {
      w <- tab[tab$model == m & tab$arm == arm & tab$measure == mn, ]
      o[[mn]] <- if (nrow(w)) w$value[match(o$opening, w$opening)] else NA
    }
    for (ms in c("d", "d_raw")) { x <- ec_open(m, arm, ms); o[[paste0("X_", ms)]] <- if (is.null(x)) NA else x[o$opening] }
    O[[length(O) + 1]] <- o
  }
  if (length(O)) write.csv(do.call(rbind, O), file.path(out, "openings.csv"), row.names = FALSE)

  # ---- per-vector table, linearity, mass by subject and class -----------------------------------------
  v <- tab[tab$measure %in% c("d", "d_raw", "fit_a", "fit_c", "ref_lo", "ref_hi", "s0_you", "s0_lm"), ]
  if (nrow(v)) {
    po <- aggregate(value ~ model + arm + dose + vec + vclass + measure + opening, v, mean)
    V <- aggregate(value ~ model + arm + dose + vec + vclass + measure, po, function(z) c(mean = mean(z), sd = sd(z), min = min(z), max = max(z), n = length(z)))
    V <- do.call(data.frame, V)
    write.csv(V, file.path(out, "vectors.csv"), row.names = FALSE)
  }
  l <- tab[tab$measure == "ref_resid", ]
  if (nrow(l)) write.csv(do.call(data.frame, aggregate(value ~ model + arm + vclass + para, l, function(z) c(mean = mean(z), sd = sd(z), n = length(z)))),
                         file.path(out, "linearity.csv"), row.names = FALSE)
  ms <- tab[grepl("^(persona_)?(mass|mdrop)_", tab$measure), ]
  if (nrow(ms)) write.csv(aggregate(value ~ model + arm + vclass + measure, ms, mean), file.path(out, "mass.csv"), row.names = FALSE)
  fl <- tab[grepl("^flip_", tab$measure) | tab$measure == "d_flip", ]
  if (nrow(fl)) write.csv(aggregate(value ~ model + arm + dose + vclass + measure, fl, mean), file.path(out, "flips.csv"), row.names = FALSE)
  if (length(KR)) write.csv(do.call(rbind, KR), file.path(out, "kr.csv"), row.names = FALSE)
  write.csv(do.call(rbind, FALLBACKS), file.path(out, "fallbacks.csv"), row.names = FALSE)

  show <- H[, c("id", "model", "arm", "estimate", "ci95_lo", "ci95_hi", "df", "p_model", "p_perm", "p", "alpha", "significant", "gate_open", "equivalent", "fit")]
  print(format(show, digits = 3), row.names = FALSE)
  print(hd[, c("model", "s2_h1", "headline")], row.names = FALSE)
}

if (!interactive()) main(commandArgs(trailingOnly = TRUE))
