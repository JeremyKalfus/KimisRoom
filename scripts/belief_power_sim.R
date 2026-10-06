# Power for the primary tests of the belief replication (drafts/prereg-amendment-belief.md): emotions -
# controls on `belief`, one model, arm D1.4, primary dose, registered maximal model, two-sided alpha
# 0.025 (the stricter Holm step for two primary tests). Same simulation as scripts/power_sim.R.
#
# Variance components are the maximal-model fits of exactly that contrast on the original confirmatory
# data (results/confirm/tables/{qwen,k26}.csv, measure belief, 40 tasks x 10 paraphrases), the data the
# hypothesis came from. SDs: task intercept, task slope, paraphrase intercept, paraphrase slope,
# residual. The effect is the observed estimate shrunk by half (Qwen -0.149 -> -0.074, K2.6 -0.071 ->
# -0.036); `full` rows use the unshrunk estimate. infl multiplies the task and paraphrase SDs (1.5 is
# the pessimistic scenario of PREREG.md section 3). P = 10 is the registered design (the same 10
# belief wordings); P = 20 and 30 are shown only to locate the paraphrase-slope ceiling.
#   Rscript scripts/belief_power_sim.R   # about 40 min on 10 cores
suppressMessages({library(lmerTest); library(parallel)})
set.seed(20261006)
VC <- list(qwen = c(ti = 0.090, ts = 0.070, pi = 0.058, ps = 0.038, e = 0.184, nE = 6, obs = -0.1486),
           k26  = c(ti = 0.093, ts = 0.028, pi = 0.048, ps = 0.061, e = 0.220, nE = 4, obs = -0.0711))
sim1 <- function(v, T, P, beta, infl) {
  nE <- v[["nE"]]; nC <- 4; vecs <- c(paste0("e", 1:nE), paste0("c", 1:nC))
  d <- expand.grid(task = 1:T, para = 1:P, vec = vecs); d$E <- as.numeric(substr(d$vec, 1, 1) == "e")
  ti <- rnorm(T, 0, v[["ti"]] * infl); ts <- rnorm(T, 0, v[["ts"]] * infl)
  pi <- rnorm(P, 0, v[["pi"]]); ps <- rnorm(P, 0, v[["ps"]] * infl)
  d$value <- ti[d$task] + pi[d$para] + d$E * (beta + ts[d$task] + ps[d$para]) + rnorm(nrow(d), 0, v[["e"]])
  d$task <- factor(d$task); d$para <- factor(d$para)
  f <- suppressMessages(suppressWarnings(lmer(value ~ E + (1 + E | task) + (1 + E | para), data = d,
         control = lmerControl(check.conv.singular = "ignore", calc.derivs = FALSE))))
  co <- summary(f)$coefficients["E", ]
  c(p = co[[5]], est = co[[1]], se = co[[2]], df = co[[3]])
}
grid <- rbind(
  expand.grid(model = c("qwen", "k26"), T = c(20, 30, 40, 60, 80), P = 10, infl = c(1, 1.5), effect = "half", stringsAsFactors = FALSE),
  expand.grid(model = c("qwen", "k26"), T = 40, P = 10, infl = c(1, 1.5), effect = "full", stringsAsFactors = FALSE),
  expand.grid(model = "k26", T = 40, P = c(20, 30), infl = c(1, 1.5), effect = "half", stringsAsFactors = FALSE))
NS <- 300
out <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  g <- grid[i, ]; v <- VC[[g$model]]; beta <- v[["obs"]] * (if (g$effect == "half") 0.5 else 1)
  r1 <- do.call(rbind, mclapply(1:NS, function(s) sim1(v, g$T, g$P, beta, g$infl), mc.cores = 10))
  r0 <- do.call(rbind, mclapply(1:NS, function(s) sim1(v, g$T, g$P, 0, g$infl), mc.cores = 10))
  r <- data.frame(g, beta = beta, power_025 = mean(r1[, "p"] < 0.025 & r1[, "est"] < 0), power_05 = mean(r1[, "p"] < 0.05 & r1[, "est"] < 0),
                  mean_se = mean(r1[, "se"]), mean_df = mean(r1[, "df"]), typeI_05 = mean(r0[, "p"] < 0.05))
  print(r, digits = 3, row.names = FALSE); r
}))
print(out, digits = 3, row.names = FALSE)
write.csv(out, "results/prereg/belief_power_sim.csv", row.names = FALSE)
