# Confirmatory analysis for amendment 1, the Qwen3 size sweep (drafts/prereg-amendment-sweep.md).
# To be frozen with the amendment. Reuses runs/confirm.R's model fitting, fallback order, data
# selections and SESOI unchanged (sourced, not copied); only the per-size constants and the tests differ.
#
#   Rscript runs/sweep_confirm.R out_dir q0.6b.csv q1.7b.csv q4b.csv q8b.csv q14b.csv
#
# Tables come from runs/confirm_table.py with the model label set to q0.6b, q1.7b, q4b, q8b or q14b
# and --entity Qwen. Writes out_dir/hypotheses.csv (Holm within family), out_dir/scaling.csv (the
# size trends) and out_dir/robustness.csv.

suppressMessages(library(lmerTest))
for (e in head(parse("runs/confirm.R"), -1)) eval(e, globalenv())  # everything but confirm.R's main() call

# ---- registered constants: the selection rule's output on the dev pilot (runs/sweep_summary.py) ----
SIX <- c("joy", "sadness", "anger", "fear", "euphoria", "surprise")
CFG <- list(
  q0.6b = list(layer = 18, dose = 2, emo = SIX, params = 0.6e9),
  q1.7b = list(layer = NA, dose = 2, emo = SIX, params = 1.7e9),   # filled from results/sweep/qwen3-1.7b/choice.json
  q4b   = list(layer = NA, dose = 2, emo = SIX, params = 4.0e9),   # filled from results/sweep/qwen3-4b/choice.json
  q8b   = list(layer = 17, dose = 2, emo = SIX, params = 8.2e9),
  q14b  = list(layer = 19, dose = 2, emo = SIX, params = 14.8e9))
SIZES <- names(CFG)
SEM <- c("red", "ocean")  # the semantic controls (robustness of the belief test)

# emotions minus a chosen control set, on one measure (confirm.R's emo_vs_ctrl with a control argument)
emo_vs <- function(model, arm, measure, ctrl) {
  cfg <- CFG[[model]]
  d <- sel(model, arm, measure, c(cfg$emo, ctrl))
  if (!nrow(d)) return(NULL)
  data.frame(task = factor(d$task), para = factor(d$para), y = d$value, x = as.numeric(d$vec %in% cfg$emo))
}

# Size trend: inverse-variance weighted regression of the per-size estimates on log10(parameters),
# fixed effect (the claim is about these five checkpoints), with a DerSimonian-Laird random-effects
# version reported beside it. Two-sided z test of the slope.
trend <- function(H, ids, what, predicted) {
  r <- H[H$id %in% ids & !is.na(H$estimate), ]
  x <- log10(sapply(r$model, function(m) CFG[[m]]$params)); y <- r$estimate; v <- r$se^2
  fe <- function(w) {
    xb <- sum(w * x) / sum(w); b <- sum(w * (x - xb) * (y - sum(w * y) / sum(w))) / sum(w * (x - xb)^2)
    c(slope = b, se = sqrt(1 / sum(w * (x - xb)^2)))
  }
  f <- fe(1 / v)
  a <- sum((1 / v) * x) / sum(1 / v); res <- y - (sum(y / v) / sum(1 / v) + f[["slope"]] * (x - a))
  Q <- sum(res^2 / v); w <- 1 / v; k <- length(y)
  tau2 <- max(0, (Q - (k - 2)) / (sum(w) - sum(w^2 * (x - a)^2) / sum(w * (x - a)^2) - sum(w^2) / sum(w)))
  re <- fe(1 / (v + tau2))
  z <- f[["slope"]] / f[["se"]]
  data.frame(what = what, predicted = predicted, n_sizes = k, slope_per_decade = f[["slope"]], se = f[["se"]],
             z = z, p = 2 * pnorm(-abs(z)), ci95_lo = f[["slope"]] - 1.96 * f[["se"]], ci95_hi = f[["slope"]] + 1.96 * f[["se"]],
             Q = Q, tau2 = tau2, re_slope = re[["slope"]], re_se = re[["se"]],
             supported = 2 * pnorm(-abs(z)) < 0.05 && sign(f[["slope"]]) == ifelse(predicted == "+", 1, -1))
}

main <- function(args) {
  out <- args[1]
  tab <<- do.call(rbind, lapply(args[-1], read.csv, stringsAsFactors = FALSE))
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  H <- list()
  add <- function(...) H[[length(H) + 1]] <<- run(...)
  for (m in SIZES) {
    add(paste0("H1.", m), "primary", m, "D1.4: self_resid, emotions - controls", emo_vs_ctrl(m, "D1.4"))
    add(paste0("H2.", m), "secondary", m, "D0: self_resid, emotions - controls", emo_vs_ctrl(m, "D0"))
    add(paste0("H3.", m), "secondary", m, "contrast under D1.4 minus under D0",
        contrast_diff(contrast_cells(m, "D0"), contrast_cells(m, "D1.4")), predicted = "-")
    add(paste0("H4.", m), "secondary", m, "D1.4: identification, self - user (emotions)",
        margin_cells(m, "D1.4", "id_self", "id_user"), coef = "(Intercept)")
    add(paste0("H6.", m), "secondary", m, "D1.4: who_felt_it, answer site - task site (emotions)",
        margin_cells(m, "D1.4", "who", "who", "answer", "task"), coef = "(Intercept)")
    add(paste0("H7.", m), "secondary", m, "D1.4: belief, emotions - controls", emo_vs_ctrl(m, "D1.4", "belief"), predicted = "-")
    add(paste0("H8.", m), "secondary", m, "D0: belief, emotions - controls", emo_vs_ctrl(m, "D0", "belief"), predicted = "-")
  }
  H <- do.call(rbind, H)
  H$p_holm <- NA
  for (f in unique(H$family)) {
    i <- H$family == f & !is.na(H$p)
    H$p_holm[i] <- p.adjust(H$p[i], "holm")
  }
  H$supported <- with(H, !is.na(p_holm) & p_holm < 0.05 & sign(estimate) == ifelse(predicted == "+", 1, -1))
  write.csv(H, file.path(out, "hypotheses.csv"), row.names = FALSE)

  S <- rbind(trend(H, paste0("H2.", SIZES), "S1: D0 contrast on log10 parameters", "+"),
             trend(H, paste0("H1.", SIZES), "S2: D1.4 contrast on log10 parameters", "+"),
             trend(H, paste0("H7.", SIZES), "S3 (exploratory): D1.4 belief contrast on log10 parameters", "-"))
  S$p_holm <- c(p.adjust(S$p[1:2], "holm"), NA)
  write.csv(S, file.path(out, "scaling.csv"), row.names = FALSE)

  R <- list()
  for (m in SIZES) for (a in c("D1.4", "D0")) {
    R[[length(R) + 1]] <- cbind(data.frame(model = m, arm = a, what = "belief, emotions - red+ocean"),
                                (function(d) if (is.null(d)) na_row() else fit_test(d))(emo_vs(m, a, "belief", SEM)))
    R[[length(R) + 1]] <- cbind(data.frame(model = m, arm = a, what = "self_resid, emotions - red+ocean"),
                                (function(d) if (is.null(d)) na_row() else fit_test(d))(emo_vs(m, a, "self_resid", SEM)))
  }
  write.csv(do.call(rbind, R), file.path(out, "robustness.csv"), row.names = FALSE)
  show <- H[, c("id", "what", "estimate", "ci95_lo", "ci95_hi", "p_holm", "equivalent", "supported", "fit", "n_tasks", "n_para")]
  print(format(show, digits = 3), row.names = FALSE)
  print(format(S, digits = 3), row.names = FALSE)
}

if (!interactive()) main(commandArgs(trailingOnly = TRUE))
