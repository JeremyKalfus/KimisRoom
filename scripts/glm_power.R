# Power for the GLM-5.3 tests (drafts/prereg-amendment-glm.md, section 3): the simulation of
# scripts/power_sim.R (prereg-v1), run on GLM's own pilot variance components.
#
#   Rscript scripts/glm_power.R table.csv LAYER DOSE EMOTIONS out.csv [NS]
#
# table.csv is runs/confirm_table.py on the GLM pilot (dev tasks, --entity GLM); LAYER, DOSE (primary,
# natural units) and EMOTIONS (comma-separated) are the selection rule's output (runs/glm_summary.py
# cal). The variance components come from the registered maximal model fit to the pilot at D1.4, the
# chosen layer and the primary dose, exactly as for prereg-v1 (task and paraphrase intercept and
# slope SDs, residual SD). Cells:
#   H1g    self_resid, 40 tasks x 30 self-state paraphrases (items_confirm), effect +0.10 (SESOI) and 0
#   H7g.1  belief, 40 tasks x the 10 belief_revision wordings, effect -0.10 and 0
# each with the pilot SDs and the pessimistic scenario (task SDs and slope SDs x 1.5, as prereg-v1).
# Power is reported at alpha 0.05 (H1g is alone in its family), 0.025 (prereg-v1's step, for
# comparison) and 0.00625 (the strictest Holm step of the eight secondary tests).
suppressMessages({library(lmerTest); library(parallel)})
set.seed(20261006)
args <- commandArgs(TRUE)
CTRL <- c("red", "ocean", "random", "arithmetic")
tab <- read.csv(args[1], stringsAsFactors = FALSE)
LAYER <- as.integer(args[2]); DOSE <- as.numeric(args[3]); EMO <- strsplit(args[4], ",")[[1]]
out_csv <- args[5]
NS <- if (length(args) >= 6) as.integer(args[6]) else 300

vc_of <- function(measure) {
  d <- tab[tab$measure == measure & tab$arm == "D1.4" & tab$site == "answer" & tab$layer == LAYER &
           abs(tab$dose - DOSE) < 1e-9 & tab$vec %in% c(EMO, CTRL), ]
  stopifnot(nrow(d) > 0, any(d$vec %in% EMO), any(d$vec %in% CTRL))
  d$E <- as.numeric(d$vec %in% EMO); d$task <- factor(d$task); d$para <- factor(d$para)
  f <- suppressMessages(suppressWarnings(lmer(value ~ E + (1 + E | task) + (1 + E | para), data = d,
         control = lmerControl(check.conv.singular = "ignore"))))
  vc <- as.data.frame(VarCorr(f))
  g <- function(grp, v1) { s <- vc$sdcor[vc$grp == grp & vc$var1 == v1 & is.na(vc$var2)]; if (length(s)) s else 0 }
  c(ti = g("task", "(Intercept)"), ts = g("task", "E"), pi = g("para", "(Intercept)"), ps = g("para", "E"),
    e = sigma(f), nE = length(EMO), est = unname(fixef(f)["E"]), n_tasks = nlevels(d$task), n_para = nlevels(d$para))
}

# scripts/power_sim.R's sim1, unchanged: simulate, fit the registered maximal model, test E
sim1 <- function(v, T, P, beta, infl) {
  nE <- v[["nE"]]; nC <- 4; vecs <- c(paste0("e", 1:nE), paste0("c", 1:nC))
  d <- expand.grid(task=1:T, para=1:P, vec=vecs); d$E <- as.numeric(substr(d$vec,1,1)=="e")
  ti <- rnorm(T, 0, v[["ti"]]*infl); ts <- rnorm(T, 0, v[["ts"]]*infl)
  pi <- rnorm(P, 0, v[["pi"]]); ps <- rnorm(P, 0, v[["ps"]]*infl)
  d$value <- ti[d$task] + pi[d$para] + d$E*(beta + ts[d$task] + ps[d$para]) + rnorm(nrow(d), 0, v[["e"]])
  d$task <- factor(d$task); d$para <- factor(d$para)
  f <- suppressMessages(suppressWarnings(lmer(value ~ E + (1+E|task) + (1+E|para), data=d,
         control=lmerControl(check.conv.singular="ignore", calc.derivs=FALSE))))
  co <- summary(f)$coefficients["E", ]
  ci90 <- co[1] + c(-1,1)*qt(0.95, co[3])*co[2]
  c(p=co[[5]], est=co[[1]], se=co[[2]], df=co[[3]], equiv=as.numeric(ci90[1] > -0.10 & ci90[2] < 0.10))
}

cells <- list(list(test = "H1g", measure = "self_resid", P = 30, beta = 0.10),
              list(test = "H7g.1", measure = "belief", P = 10, beta = -0.10))
rows <- list()
for (cl in cells) {
  v <- vc_of(cl$measure)
  cat(sprintf("%s (%s, D1.4, layer %d, dose %g): pilot estimate %+.3f on %d tasks x %d paraphrases; SDs task %.3f/%.3f para %.3f/%.3f resid %.3f\n",
              cl$test, cl$measure, LAYER, DOSE, v[["est"]], v[["n_tasks"]], v[["n_para"]], v[["ti"]], v[["ts"]],
              v[["pi"]], v[["ps"]], v[["e"]]))
  for (infl in c(1, 1.5)) {
    r1 <- do.call(rbind, mclapply(1:NS, function(s) sim1(v, 40, cl$P, cl$beta, infl), mc.cores = 8))
    r0 <- do.call(rbind, mclapply(1:NS, function(s) sim1(v, 40, cl$P, 0, infl), mc.cores = 8))
    pw <- function(a) mean(r1[, "p"] < a & sign(r1[, "est"]) == sign(cl$beta))
    rows[[length(rows) + 1]] <- data.frame(test = cl$test, measure = cl$measure, T = 40, P = cl$P, infl = infl,
      beta = cl$beta, pilot_est = v[["est"]], ti = v[["ti"]], ts = v[["ts"]], pi = v[["pi"]], ps = v[["ps"]],
      e = v[["e"]], nE = v[["nE"]], power_05 = pw(0.05), power_025 = pw(0.025), power_00625 = pw(0.00625),
      mean_se = mean(r1[, "se"]), typeI_05 = mean(r0[, "p"] < 0.05), equiv_power = mean(r0[, "equiv"]), nsim = NS)
    print(tail(rows, 1)[[1]], digits = 3, row.names = FALSE)
  }
}
write.csv(do.call(rbind, rows), out_csv, row.names = FALSE)
