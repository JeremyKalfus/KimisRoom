# Power for the size sweep's per-size primary test (drafts/prereg-amendment-sweep.md), the same
# simulation as scripts/power_sim.R with each size's own pilot variance components: emotions - controls
# on self_resid under D1.4 at the size's primary dose, 40 tasks x 30 paraphrases, SESOI +0.10,
# maximal model, also at +0.20, alpha 0.01 (the strictest Holm step over five sizes) and 0.05.
#   Rscript runs/sweep_power.R out.csv size1.csv,emo1,layer1,dose1 [size2.csv,...]
# Each size argument is a runs/confirm_table.py table of the sweep pilot, the emotion set joined by
# '+', the rule's layer and the primary dose.
suppressMessages({library(lmerTest); library(parallel)})
set.seed(20261006)
args <- commandArgs(TRUE)
out_csv <- args[1]
CTRL <- c("red", "ocean", "random", "arithmetic")

vc_of <- function(path, emo, layer, dose) {
  d <- read.csv(path)
  d <- d[d$measure == "self_resid" & d$arm == "D1.4" & d$site == "answer" & d$layer == layer & d$dose == dose &
         d$vec %in% c(emo, CTRL), ]
  d$x <- as.numeric(d$vec %in% emo); d$task <- factor(d$task); d$para <- factor(d$para)
  f <- suppressMessages(suppressWarnings(lmer(value ~ x + (1 + x | task) + (1 + x | para), data = d,
         control = lmerControl(check.conv.singular = "ignore"))))
  vc <- as.data.frame(VarCorr(f))
  g <- function(grp, v1) { s <- vc$sdcor[vc$grp == grp & vc$var1 == v1 & is.na(vc$var2)]; if (length(s)) s else 0 }
  c(ti = g("task", "(Intercept)"), ts = g("task", "x"), pi = g("para", "(Intercept)"), ps = g("para", "x"),
    e = sigma(f), nE = length(emo), est = unname(fixef(f)["x"]))
}

sim1 <- function(v, T, P, beta, infl) {
  nE <- v[["nE"]]; vecs <- c(paste0("e", 1:nE), paste0("c", 1:4))
  d <- expand.grid(task = 1:T, para = 1:P, vec = vecs); d$E <- as.numeric(substr(d$vec, 1, 1) == "e")
  ti <- rnorm(T, 0, v[["ti"]] * infl); ts <- rnorm(T, 0, v[["ts"]] * infl)
  pi <- rnorm(P, 0, v[["pi"]]); ps <- rnorm(P, 0, v[["ps"]] * infl)
  d$value <- ti[d$task] + pi[d$para] + d$E * (beta + ts[d$task] + ps[d$para]) + rnorm(nrow(d), 0, v[["e"]])
  d$task <- factor(d$task); d$para <- factor(d$para)
  f <- suppressMessages(suppressWarnings(lmer(value ~ E + (1 + E | task) + (1 + E | para), data = d,
         control = lmerControl(check.conv.singular = "ignore", calc.derivs = FALSE))))
  co <- summary(f)$coefficients["E", ]
  c(p = co[[5]], se = co[[2]])
}

NS <- 200
rows <- list()
for (a in args[-1]) {
  p <- strsplit(a, ",")[[1]]
  v <- vc_of(p[1], strsplit(p[2], "\\+")[[1]], as.integer(p[3]), as.numeric(p[4]))
  for (infl in c(1, 1.5)) for (beta in c(0.10, 0.20)) {
    r <- do.call(rbind, mclapply(1:NS, function(s) sim1(v, 40, 30, beta, infl), mc.cores = 8))
    rows[[length(rows) + 1]] <- data.frame(table = basename(dirname(p[1])), infl = infl, beta = beta, pilot_est = v[["est"]],
      ti = v[["ti"]], ts = v[["ts"]], pi = v[["pi"]], ps = v[["ps"]], e = v[["e"]],
      power_01 = mean(r[, "p"] < 0.01), power_05 = mean(r[, "p"] < 0.05), mean_se = mean(r[, "se"]))
    print(tail(rows, 1)[[1]], digits = 3, row.names = FALSE)
  }
}
write.csv(do.call(rbind, rows), out_csv, row.names = FALSE)
