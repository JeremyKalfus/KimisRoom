# Preregistered confirmatory analysis (PREREG.md, Analysis plan). Frozen with the registration.
#
#   Rscript runs/confirm.R out_dir table1.csv [table2.csv ...]
#
# The tables come from runs/confirm_table.py, one per result directory, with the model label set to
# "qwen" or "k26" for the base models and "qwen-D2" / "qwen-D2c" for the fine-tuned Qwen copies.
# Writes out_dir/hypotheses.csv (one row per registered test, Holm-adjusted within its family) and
# out_dir/vectors.csv (per-vector estimates, descriptive). Missing data for a test gives an NA row.
# Needs R >= 4.3 with lme4 and lmerTest.

suppressMessages(library(lmerTest))

# ---- registered constants --------------------------------------------------------------------------
CTRL <- c("red", "ocean", "random", "arithmetic")
CFG <- list(
  qwen = list(layer = 40, dose = 2,  # primary dose in natural units; layer 40 of 64
              emo = c("joy", "sadness", "anger", "fear", "euphoria", "surprise")),
  k26  = list(layer = 20, dose = 4,  # layer 20 of 61; only these four pass AIPsy-Affect there
              emo = c("joy", "fear", "euphoria", "surprise")))
SESOI <- 0.10  # smallest effect of interest, log-odds; equivalence bounds are +-SESOI

# ---- model fitting with the registered fallback order ---------------------------------------------
# Maximal crossed random effects for the tested contrast x; on a convergence failure (a singular fit
# alone is not one) refit without random-effect correlations, then without the paraphrase slope,
# then without the task slope. The step used is reported.
FORMULAS <- list(
  maximal       = "y ~ x + (1 + x | task) + (1 + x | para)",
  uncorrelated  = "y ~ x + (1 + x || task) + (1 + x || para)",
  no_para_slope = "y ~ x + (1 + x || task) + (1 | para)",
  intercepts    = "y ~ x + (1 | task) + (1 | para)")
FORMULAS1 <- list(maximal = "y ~ 1 + (1 | task) + (1 | para)")  # intercept-only tests

fit_test <- function(d, coef = "x") {
  forms <- if (coef == "(Intercept)") FORMULAS1 else FORMULAS
  for (step in names(forms)) {
    warn <- NULL
    f <- withCallingHandlers(
      suppressMessages(lmer(as.formula(forms[[step]]), data = d,
                            control = lmerControl(check.conv.singular = "ignore"))),
      warning = function(w) { warn <<- conditionMessage(w); invokeRestart("muffleWarning") })
    if (is.null(warn) || step == tail(names(forms), 1)) break
  }
  co <- summary(f)$coefficients[coef, ]
  ci <- function(level) co[["Estimate"]] + c(-1, 1) * qt(1 - (1 - level) / 2, co[["df"]]) * co[["Std. Error"]]
  c95 <- ci(0.95); c90 <- ci(0.90)
  data.frame(estimate = co[["Estimate"]], se = co[["Std. Error"]], df = co[["df"]], t = co[["t value"]],
             p = co[["Pr(>|t|)"]], ci95_lo = c95[1], ci95_hi = c95[2], ci90_lo = c90[1], ci90_hi = c90[2],
             equivalent = c90[1] > -SESOI & c90[2] < SESOI, fit = step, singular = isSingular(f),
             converged = is.null(warn), n_tasks = length(unique(d$task)), n_para = length(unique(d$para)))
}

na_row <- function() data.frame(estimate = NA, se = NA, df = NA, t = NA, p = NA, ci95_lo = NA, ci95_hi = NA,
                                ci90_lo = NA, ci90_hi = NA, equivalent = NA, fit = "no data", singular = NA,
                                converged = NA, n_tasks = 0, n_para = 0)

# ---- data selections ------------------------------------------------------------------------------
tab <- NULL
sel <- function(model, arm, measure, vecs, site = "answer") {
  key <- sub("-.*", "", model)  # qwen-D2 -> qwen
  cfg <- CFG[[key]]
  tab[tab$model == model & tab$arm == arm & tab$measure == measure & tab$site == site &
      tab$layer == cfg$layer & abs(tab$dose - cfg$dose) < 1e-9 & tab$vec %in% vecs, ]
}

# emotions minus controls on one measure: rows per (task, para, vector), x = 1 for emotions
emo_vs_ctrl <- function(model, arm, measure = "self_resid") {
  cfg <- CFG[[sub("-.*", "", model)]]
  d <- sel(model, arm, measure, c(cfg$emo, CTRL))
  if (!nrow(d)) return(NULL)
  data.frame(task = factor(d$task), para = factor(d$para), y = d$value, x = as.numeric(d$vec %in% cfg$emo))
}

# the emotions-minus-controls contrast collapsed to one number per (task, para)
contrast_cells <- function(model, arm, measure = "self_resid") {
  d <- emo_vs_ctrl(model, arm, measure)
  if (is.null(d)) return(NULL)
  e <- aggregate(y ~ task + para, d[d$x == 1, ], mean); c <- aggregate(y ~ task + para, d[d$x == 0, ], mean)
  m <- merge(e, c, by = c("task", "para"), suffixes = c(".e", ".c"))
  data.frame(task = m$task, para = m$para, y = m$y.e - m$y.c)
}

# two arms (or two models) of the contrast, x = 1 for the second
contrast_diff <- function(a, b) {
  if (is.null(a) || is.null(b)) return(NULL)
  rbind(transform(a, x = 0), transform(b, x = 1))
}

# per (task, para): mean over the model's emotions of measure1 - measure2
margin_cells <- function(model, arm, m1, m2, site1 = "answer", site2 = "answer") {
  cfg <- CFG[[sub("-.*", "", model)]]
  a <- sel(model, arm, m1, cfg$emo, site1); b <- sel(model, arm, m2, cfg$emo, site2)
  if (!nrow(a) || !nrow(b)) return(NULL)
  m <- merge(a, b, by = c("task", "para", "vec"), suffixes = c(".1", ".2"))
  m$y <- m$value.1 - m$value.2
  g <- aggregate(y ~ task + para, m, mean)
  data.frame(task = factor(g$task), para = factor(g$para), y = g$y)
}

run <- function(id, family, model, what, d, coef = "x", predicted = "+") {
  r <- if (is.null(d) || !nrow(d)) na_row() else fit_test(d, coef)
  cbind(data.frame(id = id, family = family, model = model, what = what, predicted = predicted), r)
}

main <- function(args) {
  out <- args[1]
  tab <<- do.call(rbind, lapply(args[-1], read.csv, stringsAsFactors = FALSE))
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  H <- list()
  for (m in c("qwen", "k26")) {
    k <- if (m == "qwen") "q" else "k"
    H[[length(H) + 1]] <- run(paste0("H1", k), "primary", m,
      "D1.4: self_resid, emotions - controls", emo_vs_ctrl(m, "D1.4"))
    H[[length(H) + 1]] <- run(paste0("H2", k), "secondary", m,
      "D0: self_resid, emotions - controls", emo_vs_ctrl(m, "D0"))
    H[[length(H) + 1]] <- run(paste0("H3", k), "secondary", m,
      "contrast under D1.4 minus under D0", contrast_diff(contrast_cells(m, "D0"), contrast_cells(m, "D1.4")),
      predicted = "-")
    H[[length(H) + 1]] <- run(paste0("H4", k), "secondary", m,
      "D1.4: identification, self - user (emotions)", margin_cells(m, "D1.4", "id_self", "id_user"),
      coef = "(Intercept)")
    H[[length(H) + 1]] <- run(paste0("H6", k), "secondary", m,
      "D1.4: who_felt_it, answer site - task site (emotions)",
      margin_cells(m, "D1.4", "who", "who", "answer", "task"), coef = "(Intercept)")
  }
  H[[length(H) + 1]] <- run("H5q", "secondary", "qwen-D2 vs qwen-D2c",
    "D0 conditioning: contrast on D2 minus on D2c",
    contrast_diff(contrast_cells("qwen-D2c", "D0"), contrast_cells("qwen-D2", "D0")), predicted = "-")
  H <- do.call(rbind, H)
  H$p_holm <- NA
  for (f in unique(H$family)) {
    i <- H$family == f & !is.na(H$p)
    H$p_holm[i] <- p.adjust(H$p[i], "holm")
  }
  H$supported <- with(H, !is.na(p_holm) & p_holm < 0.05 & sign(estimate) == ifelse(predicted == "+", 1, -1))
  write.csv(H, file.path(out, "hypotheses.csv"), row.names = FALSE)

  # robustness for H1/H2: per-task t-tests of the contrast, on self_resid and on self_raw - other_raw
  ttest <- function(per_task) {
    if (length(per_task) < 2) return(c(estimate = NA, ci95_lo = NA, ci95_hi = NA, p = NA, n_tasks = length(per_task)))
    tt <- t.test(per_task)
    c(estimate = unname(tt$estimate), ci95_lo = tt$conf.int[1], ci95_hi = tt$conf.int[2], p = tt$p.value, n_tasks = length(per_task))
  }
  task_contrast <- function(d) {  # d: task, x, y -> emotions minus controls per task
    if (is.null(d) || !nrow(d)) return(numeric(0))
    e <- tapply(d$y[d$x == 1], d$task[d$x == 1], mean); c <- tapply(d$y[d$x == 0], d$task[d$x == 0], mean)
    k <- intersect(names(e), names(c)); e[k] - c[k]
  }
  raw_diff <- function(model, arm) {  # per (task, vector): mean self_raw - mean other_raw
    cfg <- CFG[[sub("-.*", "", model)]]
    a <- sel(model, arm, "self_raw", c(cfg$emo, CTRL)); b <- sel(model, arm, "other_raw", c(cfg$emo, CTRL))
    if (!nrow(a) || !nrow(b)) return(NULL)
    a <- aggregate(value ~ task + vec, a, mean); b <- aggregate(value ~ task + vec, b, mean)
    m <- merge(a, b, by = c("task", "vec"), suffixes = c(".s", ".o"))
    data.frame(task = m$task, x = as.numeric(m$vec %in% cfg$emo), y = m$value.s - m$value.o)
  }
  R <- list()
  for (m in c("qwen", "k26")) for (a in c("D1.4", "D0")) {
    id <- paste0(if (a == "D1.4") "H1" else "H2", if (m == "qwen") "q" else "k")
    R[[length(R) + 1]] <- data.frame(id = id, model = m, arm = a, measure = "self_resid",
                                     t(ttest(task_contrast(emo_vs_ctrl(m, a)))))
    R[[length(R) + 1]] <- data.frame(id = id, model = m, arm = a, measure = "self_raw - other_raw",
                                     t(ttest(task_contrast(raw_diff(m, a)))))
  }
  write.csv(do.call(rbind, R), file.path(out, "robustness.csv"), row.names = FALSE)

  # descriptive: every vector's mean self_resid per model x arm x dose, with the per-task SD
  v <- tab[tab$measure == "self_resid", ]
  if (nrow(v)) {
    pt <- aggregate(value ~ model + arm + site + layer + dose + vec + task, v, mean)
    V <- aggregate(value ~ model + arm + site + layer + dose + vec, pt, function(z) c(mean = mean(z), sd = sd(z), n = length(z)))
    V <- do.call(data.frame, V)
    write.csv(V, file.path(out, "vectors.csv"), row.names = FALSE)
  }
  show <- H[, c("id", "model", "what", "estimate", "ci95_lo", "ci95_hi", "p", "p_holm", "equivalent", "supported", "fit", "n_tasks", "n_para")]
  print(format(show, digits = 3), row.names = FALSE)
}

if (!interactive()) main(commandArgs(trailingOnly = TRUE))
