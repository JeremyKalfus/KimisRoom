# GLM-5.3 tests (drafts/prereg-amendment-glm.md), on the frozen model code of runs/confirm.R.
#
#   Rscript runs/glm_confirm.R out_dir LAYER DOSE EMOTIONS table.csv [model_label]
#
# LAYER, DOSE (primary, natural units) and EMOTIONS (comma-separated) are the selection rule's output
# (runs/glm_summary.py cal). The table comes from runs/confirm_table.py with --entity GLM; its model
# label defaults to "glm". Writes out_dir/hypotheses.csv, one row per test, Holm-adjusted within its
# family: H1g alone in the primary family; H2g, H3g, H4g, H6g and the four general-belief tests H7g
# in the secondary family. The models, fallback order, SESOI and the support rule are confirm.R's.
# On pilot data (dev tasks) these are effect sizes for planning, not tests.

suppressMessages(library(lmerTest))
for (e in head(parse("runs/confirm.R"), -1)) eval(e, globalenv())  # everything but the main() call

SEM <- c("red", "ocean")

# emotions minus a given control set on one measure, rows per (task, para, vector), x = 1 for emotions
emo_vs <- function(model, arm, measure, ctrl) {
  cfg <- CFG[[sub("-.*", "", model)]]
  d <- sel(model, arm, measure, c(cfg$emo, ctrl))
  if (!nrow(d) || !any(d$vec %in% cfg$emo) || !any(d$vec %in% ctrl)) return(NULL)
  data.frame(task = factor(d$task), para = factor(d$para), y = d$value, x = as.numeric(d$vec %in% cfg$emo))
}

main_glm <- function(args) {
  out <- args[1]
  label <- if (length(args) >= 6) args[6] else "glm"
  CFG[[sub("-.*", "", label)]] <<- list(layer = as.integer(args[2]), dose = as.numeric(args[3]),
                        emo = strsplit(args[4], ",")[[1]])
  tab <<- read.csv(args[5], stringsAsFactors = FALSE)
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  m <- label
  H <- list(
    run("H1g", "primary", m, "D1.4: self_resid, emotions - controls", emo_vs_ctrl(m, "D1.4")),
    run("H2g", "secondary", m, "D0: self_resid, emotions - controls", emo_vs_ctrl(m, "D0")),
    run("H3g", "secondary", m, "contrast under D1.4 minus under D0",
        contrast_diff(contrast_cells(m, "D0"), contrast_cells(m, "D1.4")), predicted = "-"),
    run("H4g", "secondary", m, "D1.4: identification, self - user (emotions)",
        margin_cells(m, "D1.4", "id_self", "id_user"), coef = "(Intercept)"),
    run("H6g", "secondary", m, "D1.4: who_felt_it, answer site - task site (emotions)",
        margin_cells(m, "D1.4", "who", "who", "answer", "task"), coef = "(Intercept)"),
    run("H7g.1", "secondary", m, "D1.4: belief, emotions - controls", emo_vs(m, "D1.4", "belief", CTRL), predicted = "-"),
    run("H7g.2", "secondary", m, "D1.4: belief, emotions - red+ocean", emo_vs(m, "D1.4", "belief", SEM), predicted = "-"),
    run("H7g.3", "secondary", m, "D0: belief, emotions - controls", emo_vs(m, "D0", "belief", CTRL), predicted = "-"),
    run("H7g.4", "secondary", m, "D0: belief, emotions - red+ocean", emo_vs(m, "D0", "belief", SEM), predicted = "-"))
  H <- do.call(rbind, H)
  H$p_holm <- NA
  for (f in unique(H$family)) {
    i <- H$family == f & !is.na(H$p)
    H$p_holm[i] <- p.adjust(H$p[i], "holm")
  }
  H$supported <- with(H, !is.na(p_holm) & p_holm < 0.05 & sign(estimate) == ifelse(predicted == "+", 1, -1))
  write.csv(H, file.path(out, "hypotheses.csv"), row.names = FALSE)
  show <- H[, c("id", "what", "estimate", "ci95_lo", "ci95_hi", "p", "p_holm", "supported", "fit", "n_tasks", "n_para")]
  print(format(show, digits = 3), row.names = FALSE)
}

if (!interactive()) main_glm(commandArgs(trailingOnly = TRUE))
