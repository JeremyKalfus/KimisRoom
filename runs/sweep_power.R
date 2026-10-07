# Power for the size sweep's tests (drafts/prereg-amendment-sweep.md), the same simulation as
# scripts/power_sim.R with each size's own pilot variance components: emotions - controls on self_resid at
# the size's primary dose, 40 tasks x 30 paraphrases, maximal model.
#   Rscript runs/sweep_power.R out.csv size1.csv,emo1,layer1,dose1[,arm[,ctrl[,mode]]] [size2.csv,...]
# Each size argument is a runs/confirm_table.py table of the sweep pilot, the emotion set joined by
# '+', the rule's layer and the primary dose; then optionally the arm (D1.4, the default, or D0), the
# control set (all, the default: red, ocean, random, arithmetic; or concept: red, ocean) and the mode:
#   test (default)  power of the two-sided test at true effects +0.10 (SESOI) and +0.20, alpha 0.01 and 0.05
#                   (H1s with arm D1.4 and ctrl all; H8s with ctrl concept);
#   tost            power of the equivalence test within +-0.10 (two one-sided tests at 0.05, Satterthwaite
#                   df) at true effects 0 and +0.05 (H9s, arm D0).
# Variance components come from the pilot at the size's own arm and control set.
suppressMessages({library(lmerTest); library(parallel)})
set.seed(20261006)
args <- commandArgs(TRUE)
out_csv <- args[1]
CTRLS <- list(all = c("red", "ocean", "random", "arithmetic"), concept = c("red", "ocean"))
SESOI <- 0.10

vc_of <- function(path, emo, layer, dose, arm, ctrl) {
  d <- read.csv(path)
  d <- d[d$measure == "self_resid" & d$arm == arm & d$site == "answer" & d$layer == layer & d$dose == dose &
         d$vec %in% c(emo, ctrl), ]
  d$x <- as.numeric(d$vec %in% emo); d$task <- factor(d$task); d$para <- factor(d$para)
  f <- suppressMessages(suppressWarnings(lmer(value ~ x + (1 + x | task) + (1 + x | para), data = d,
         control = lmerControl(check.conv.singular = "ignore"))))
  vc <- as.data.frame(VarCorr(f))
  g <- function(grp, v1) { s <- vc$sdcor[vc$grp == grp & vc$var1 == v1 & is.na(vc$var2)]; if (length(s)) s else 0 }
  c(ti = g("task", "(Intercept)"), ts = g("task", "x"), pi = g("para", "(Intercept)"), ps = g("para", "x"),
    e = sigma(f), nE = length(emo), nC = length(ctrl), est = unname(fixef(f)["x"]))
}

sim1 <- function(v, T, P, beta, infl) {
  nE <- v[["nE"]]; vecs <- c(paste0("e", 1:nE), paste0("c", 1:v[["nC"]]))
  d <- expand.grid(task = 1:T, para = 1:P, vec = vecs); d$E <- as.numeric(substr(d$vec, 1, 1) == "e")
  ti <- rnorm(T, 0, v[["ti"]] * infl); ts <- rnorm(T, 0, v[["ts"]] * infl)
  pi <- rnorm(P, 0, v[["pi"]]); ps <- rnorm(P, 0, v[["ps"]] * infl)
  d$value <- ti[d$task] + pi[d$para] + d$E * (beta + ts[d$task] + ps[d$para]) + rnorm(nrow(d), 0, v[["e"]])
  d$task <- factor(d$task); d$para <- factor(d$para)
  f <- suppressMessages(suppressWarnings(lmer(value ~ E + (1 + E | task) + (1 + E | para), data = d,
         control = lmerControl(check.conv.singular = "ignore", calc.derivs = FALSE))))
  co <- summary(f)$coefficients["E", ]
  est <- co[[1]]; se <- co[[2]]; df <- co[[3]]
  p_tost <- max(pt((est - SESOI) / se, df), pt((est + SESOI) / se, df, lower.tail = FALSE))
  c(p = co[[5]], se = se, p_tost = p_tost)
}

NS <- 200
rows <- list()
for (a in args[-1]) {
  p <- strsplit(a, ",")[[1]]
  arm <- if (length(p) >= 5) p[5] else "D1.4"
  cname <- if (length(p) >= 6) p[6] else "all"
  mode <- if (length(p) >= 7) p[7] else "test"
  v <- vc_of(p[1], strsplit(p[2], "\\+")[[1]], as.integer(p[3]), as.numeric(p[4]), arm, CTRLS[[cname]])
  betas <- if (mode == "tost") c(0, 0.05) else c(0.10, 0.20)
  for (infl in c(1, 1.5)) for (beta in betas) {
    r <- do.call(rbind, mclapply(1:NS, function(s) sim1(v, 40, 30, beta, infl), mc.cores = 8))
    rows[[length(rows) + 1]] <- data.frame(table = basename(dirname(p[1])), arm = arm, ctrl = cname, mode = mode,
      infl = infl, beta = beta, pilot_est = v[["est"]],
      ti = v[["ti"]], ts = v[["ts"]], pi = v[["pi"]], ps = v[["ps"]], e = v[["e"]],
      power_01 = mean(r[, "p"] < 0.01), power_05 = mean(r[, "p"] < 0.05),
      power_tost = mean(r[, "p_tost"] < 0.05), mean_se = mean(r[, "se"]))
    print(tail(rows, 1)[[1]], digits = 3, row.names = FALSE)
  }
}
write.csv(do.call(rbind, rows), out_csv, row.names = FALSE)
