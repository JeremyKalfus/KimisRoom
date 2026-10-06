# Registered analysis of the belief replication (drafts/prereg-amendment-belief.md). Frozen with the
# amendment.
#
#   Rscript runs/belief_confirm.R out_dir table1.csv [table2.csv ...]
#
# The tables come from `runs/confirm_table.py <dir> <label> <out.csv> [--entity Qwen] --belief`, one per
# result directory, labels "qwen", "k26", "qwen-D2", "qwen-D2c". Reuses the frozen model code of
# runs/confirm.R unchanged (CFG, CTRL, fit_test with its fallback order, sel, emo_vs_ctrl, run), as
# runs/confirm_explore.R does, and writes:
#   out_dir/hypotheses.csv   one row per registered test, Holm-adjusted within its family
#   out_dir/robustness.csv   per-task t-tests of the B1/B2 contrasts on belief and on belief_resid
#   out_dir/explore.csv      exploratory rows (secondary dose, task site, D2c, per-control split, calm)
#   out_dir/vectors.csv      every vector's mean belief per model x arm x site x dose (descriptive)
# Needs R >= 4.3 with lme4 and lmerTest.

suppressMessages(library(lmerTest))
for (e in head(parse("runs/confirm.R"), -1)) eval(e, globalenv())  # everything but confirm.R's main() call

SEM <- c("red", "ocean"); NONSEM <- c("arithmetic", "random")
DOSE2 <- c(qwen = 1, k26 = 2)  # the secondary doses

# rows of one measure for a set of vectors at the model's registered layer (primary dose unless given)
pick <- function(model, arm, measure, vecs, site = "answer", dose = NULL) {
  cfg <- CFG[[sub("-.*", "", model)]]
  dose <- if (is.null(dose)) cfg$dose else dose
  tab[tab$model == model & tab$arm == arm & tab$measure == measure & tab$site == site &
      tab$layer == cfg$layer & abs(tab$dose - dose) < 1e-9 & tab$vec %in% vecs, ]
}
# emotions minus a control set on one measure (rows per task, para, vector; x = 1 for emotions); the
# emotion vectors sit on the answer, or on the task text with emo_site = "task" (controls: answer only)
contrast <- function(model, arm, emo, ctrl, measure, dose = NULL, emo_site = "answer") {
  e <- pick(model, arm, measure, emo, emo_site, dose); c <- pick(model, arm, measure, ctrl, "answer", dose)
  if (!nrow(e) || !nrow(c)) return(NULL)
  data.frame(task = factor(c(e$task, c$task)), para = factor(c(e$para, c$para)), y = c(e$value, c$value),
             x = rep(1:0, c(nrow(e), nrow(c))))
}

main_belief <- function(args) {
  out <- args[1]
  tab <<- do.call(rbind, lapply(args[-1], read.csv, stringsAsFactors = FALSE))
  tab <<- tab[tab$measure %in% c("belief", "belief_resid", "belief_neutral"), ]
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  H <- list(); add <- function(r) H[[length(H) + 1]] <<- r
  for (m in c("qwen", "k26")) {
    k <- if (m == "qwen") "q" else "k"; emo <- CFG[[m]]$emo
    add(run(paste0("B1", k), "primary", m, "D1.4: belief, emotions - controls", emo_vs_ctrl(m, "D1.4", "belief"), predicted = "-"))
    add(run(paste0("B2", k), "secondary", m, "D0: belief, emotions - controls", emo_vs_ctrl(m, "D0", "belief"), predicted = "-"))
    add(run(paste0("B3", k), "secondary", m, "D1.4: belief, emotions - red+ocean", contrast(m, "D1.4", emo, SEM, "belief"), predicted = "-"))
    add(run(paste0("B4", k), "secondary", m, "D1.4: belief_resid, emotions - controls", emo_vs_ctrl(m, "D1.4", "belief_resid"), predicted = "-"))
    add(run(paste0("B5", k), "secondary", m, "D1.4: belief_neutral, emotions - controls", emo_vs_ctrl(m, "D1.4", "belief_neutral"), predicted = "-"))
    add(run(paste0("B6", k), "secondary", m, "D0: belief_neutral, emotions - controls", emo_vs_ctrl(m, "D0", "belief_neutral"), predicted = "-"))
  }
  add(run("B7q", "secondary", "qwen-D2", "D2 fine-tune, D0 conditioning: belief, emotions - controls",
          emo_vs_ctrl("qwen-D2", "D0", "belief"), predicted = "-"))
  H <- do.call(rbind, H)
  H$p_holm <- NA
  for (f in unique(H$family)) {
    i <- H$family == f & !is.na(H$p)
    H$p_holm[i] <- p.adjust(H$p[i], "holm")
  }
  H$supported <- with(H, !is.na(p_holm) & p_holm < 0.05 & sign(estimate) == ifelse(predicted == "+", 1, -1))
  H$equivalent <- NULL  # confirm.R's +-0.10 equivalence bounds belong to self_resid; no equivalence claim here
  write.csv(H, file.path(out, "hypotheses.csv"), row.names = FALSE)

  # robustness (not used for inference): one-sample t-tests over per-task contrasts
  ttest <- function(per_task) {
    if (length(per_task) < 2) return(c(estimate = NA, ci95_lo = NA, ci95_hi = NA, p = NA, n_tasks = length(per_task)))
    tt <- t.test(per_task)
    c(estimate = unname(tt$estimate), ci95_lo = tt$conf.int[1], ci95_hi = tt$conf.int[2], p = tt$p.value, n_tasks = length(per_task))
  }
  task_contrast <- function(d) {
    if (is.null(d) || !nrow(d)) return(numeric(0))
    e <- tapply(d$y[d$x == 1], d$task[d$x == 1], mean); c <- tapply(d$y[d$x == 0], d$task[d$x == 0], mean)
    k <- intersect(names(e), names(c)); e[k] - c[k]
  }
  R <- list()
  for (m in c("qwen", "k26")) for (a in c("D1.4", "D0")) for (ms in c("belief", "belief_resid")) {
    id <- paste0(if (a == "D1.4") "B1" else "B2", if (m == "qwen") "q" else "k")
    R[[length(R) + 1]] <- data.frame(id = id, model = m, arm = a, measure = ms, t(ttest(task_contrast(emo_vs_ctrl(m, a, ms)))))
  }
  write.csv(do.call(rbind, R), file.path(out, "robustness.csv"), row.names = FALSE)

  # exploratory (uncorrected)
  X <- list(); addx <- function(analysis, model, arm, what, d, dose = NA, coef = "x")
    X[[length(X) + 1]] <<- cbind(data.frame(analysis = analysis, model = model, arm = arm, dose = dose, what = what),
                                 if (is.null(d) || !nrow(d)) na_row() else fit_test(d, coef))
  for (m in c("qwen", "k26")) {
    emo <- CFG[[m]]$emo
    for (a in c("D0", "D1.4")) {
      for (ms in c("belief", "belief_resid", "belief_neutral")) {
        addx("secondary dose", m, a, paste(ms, "emotions - controls"), contrast(m, a, emo, CTRL, ms, dose = DOSE2[[m]]), DOSE2[[m]])
        # the emotion vector on the user's task text instead of the answer (controls never run there)
        addx("task site", m, a, paste(ms, "emotions on task text - controls on answer"), contrast(m, a, emo, CTRL, ms, emo_site = "task"))
        addx("task site", m, a, paste(ms, "emotions on answer - emotions on task text"),
             margin_cells(m, a, ms, ms, "answer", "task"), coef = "(Intercept)")
        for (cs in list(SEM, NONSEM, "red", "ocean", "arithmetic", "random"))
          addx("control split", m, a, paste(ms, "emotions -", paste(cs, collapse = "+")), contrast(m, a, emo, cs, ms))
        addx("other emotions", m, a, paste(ms, "calm - controls"), contrast(m, a, "calm", CTRL, ms))
      }
    }
  }
  for (ms in c("belief", "belief_resid")) {
    addx("fine-tunes", "qwen-D2", "D0", paste(ms, "emotions - controls"), contrast("qwen-D2", "D0", CFG$qwen$emo, CTRL, ms))
    addx("fine-tunes", "qwen-D2c", "D0", paste(ms, "emotions - controls"), contrast("qwen-D2c", "D0", CFG$qwen$emo, CTRL, ms))
  }
  write.csv(do.call(rbind, X), file.path(out, "explore.csv"), row.names = FALSE)

  v <- tab[tab$measure %in% c("belief", "belief_resid", "belief_neutral"), ]
  pt <- aggregate(value ~ model + measure + arm + site + layer + dose + vec + task, v, mean)
  V <- do.call(data.frame, aggregate(value ~ model + measure + arm + site + layer + dose + vec, pt,
                                     function(z) c(mean = mean(z), sd = sd(z), n = length(z))))
  write.csv(V, file.path(out, "vectors.csv"), row.names = FALSE)

  show <- H[, c("id", "family", "model", "what", "estimate", "ci95_lo", "ci95_hi", "p", "p_holm", "supported", "fit", "n_tasks", "n_para")]
  print(format(show, digits = 3), row.names = FALSE)
}

if (!interactive()) main_belief(commandArgs(trailingOnly = TRUE))
