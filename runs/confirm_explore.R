# Exploratory analyses of the confirmatory data (PREREG.md section 5, "Exploratory analysis"). None of
# these carries a confirmatory claim, and p-values are uncorrected.
#
#   Rscript runs/confirm_explore.R out_dir table1.csv [table2.csv ...]
#
# Reuses the frozen model code of runs/confirm.R (fit_test with its fallback order) and writes
# out_dir/explore.csv, one row per analysis:
#   control split   emotions minus each control alone, and minus the concept controls (red, ocean)
#                   against the non-concept ones (arithmetic, random)
#   dose            H1/H2 contrasts at the secondary dose
#   other emotions  calm (both models); sadness and anger on K2.6, which fail AIPsy-Affect at L20
#   third person    the contrast on the third-person items about the model itself
#   belief          the contrast on belief revision (positive = denial held more firmly)
#   identification  id_self, id_user, id_text and their margins, emotions only, per arm
#   who_felt_it     the who margin per arm and site
#   fine-tunes      the H5 components: the contrast on D2 and on D2c, and their control split

suppressMessages(library(lmerTest))
for (e in head(parse("runs/confirm.R"), -1)) eval(e, globalenv())  # everything but the main() call

SEM <- c("red", "ocean"); NONSEM <- c("arithmetic", "random")
DOSE2 <- c(qwen = 1, k26 = 2)  # the secondary doses

pick <- function(model, arm, measure, vecs, site = "answer", dose = NULL) {
  cfg <- CFG[[sub("-.*", "", model)]]
  dose <- if (is.null(dose)) cfg$dose else dose
  tab[tab$model == model & tab$arm == arm & tab$measure == measure & tab$site == site &
      tab$layer == cfg$layer & abs(tab$dose - dose) < 1e-9 & tab$vec %in% vecs, ]
}
contrast <- function(model, arm, emo, ctrl, measure = "self_resid", dose = NULL) {
  d <- pick(model, arm, measure, c(emo, ctrl), dose = dose)
  if (!nrow(d) || !any(d$vec %in% emo) || !any(d$vec %in% ctrl)) return(NULL)
  data.frame(task = factor(d$task), para = factor(d$para), y = d$value, x = as.numeric(d$vec %in% emo))
}
mean_cells <- function(model, arm, measure, vecs, site = "answer") {
  d <- pick(model, arm, measure, vecs, site)
  if (!nrow(d)) return(NULL)
  g <- aggregate(value ~ task + para, d, mean)
  data.frame(task = factor(g$task), para = factor(g$para), y = g$value)
}
row <- function(analysis, model, arm, what, d, coef = "x", dose = NA) {
  r <- if (is.null(d) || !nrow(d)) na_row() else fit_test(d, coef)
  cbind(data.frame(analysis = analysis, model = model, arm = arm, dose = dose, what = what), r)
}

main_explore <- function(args) {
  out <- args[1]
  tab <<- do.call(rbind, lapply(args[-1], read.csv, stringsAsFactors = FALSE))
  X <- list(); add <- function(r) X[[length(X) + 1]] <<- r
  for (m in c("qwen", "k26")) {
    emo <- CFG[[m]]$emo
    for (a in c("D0", "D1.4")) {
      for (cs in list(SEM, NONSEM, "red", "ocean", "arithmetic", "random"))
        add(row("control split", m, a, paste("emotions -", paste(cs, collapse = "+")), contrast(m, a, emo, cs)))
      add(row("dose", m, a, "emotions - controls, secondary dose", contrast(m, a, emo, CTRL, dose = DOSE2[[m]]), dose = DOSE2[[m]]))
      add(row("other emotions", m, a, "calm - controls", contrast(m, a, "calm", CTRL)))
      if (m == "k26") add(row("other emotions", m, a, "sadness+anger - controls", contrast(m, a, c("sadness", "anger"), CTRL)))
      add(row("third person", m, a, "emotions - controls on tp_self_resid", contrast(m, a, emo, CTRL, "tp_self_resid")))
      add(row("belief", m, a, "emotions - controls on belief", contrast(m, a, emo, CTRL, "belief")))
      add(row("belief", m, a, "emotions - red+ocean on belief", contrast(m, a, emo, SEM, "belief")))
      for (k in c("id_self", "id_user", "id_text"))
        add(row("identification", m, a, k, mean_cells(m, a, k, emo), coef = "(Intercept)"))
      add(row("identification", m, a, "id_self - id_user", margin_cells(m, a, "id_self", "id_user"), coef = "(Intercept)"))
      add(row("identification", m, a, "id_self - id_text", margin_cells(m, a, "id_self", "id_text"), coef = "(Intercept)"))
      for (s in c("answer", "task"))
        add(row("who_felt_it", m, a, paste("who, vector on", s), mean_cells(m, a, "who", emo, s), coef = "(Intercept)"))
    }
  }
  for (m in c("qwen-D2", "qwen-D2c")) {
    emo <- CFG$qwen$emo
    add(row("fine-tunes", m, "D0", "emotions - controls", contrast(m, "D0", emo, CTRL)))
    add(row("fine-tunes", m, "D0", "emotions - red+ocean", contrast(m, "D0", emo, SEM)))
    add(row("fine-tunes", m, "D0", "emotions - arithmetic+random", contrast(m, "D0", emo, NONSEM)))
  }
  X <- do.call(rbind, X)
  write.csv(X, file.path(out, "explore.csv"), row.names = FALSE)
  print(format(X[, c("analysis", "model", "arm", "what", "estimate", "ci95_lo", "ci95_hi", "p", "fit")], digits = 3), row.names = FALSE)
}

if (!interactive()) main_explore(commandArgs(trailingOnly = TRUE))
