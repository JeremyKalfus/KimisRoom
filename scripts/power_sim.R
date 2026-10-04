# Power for the primary test H1 (PREREG.md section 3): emotions - controls in self_resid, one model,
# arm D1.4, primary dose, fit with the registered maximal model. Variance components are the pilot
# fits (results/qwen/d14 at 2 units, results/k26/L20 at 4 units, primary emotion sets, 5 tasks x 10
# paraphrases); infl multiplies the task and paraphrase SDs. Round 1 (10 and 20 paraphrases) is in
# results/prereg/power_sim_round1.csv.
#   Rscript scripts/power_sim.R   # about 30 min on 8 cores
suppressMessages({library(lmerTest); library(parallel)})
set.seed(20261004)
# pilot variance components at D1.4 and the primary dose (pilot_vc.R): SDs of
# task intercept, task slope, para intercept, para slope, residual
VC <- list(qwen=c(ti=0.032, ts=0.088, pi=0.052, ps=0.060, e=0.180, nE=6),
           k26 =c(ti=0.014, ts=0.043, pi=0.033, ps=0.089, e=0.204, nE=4))
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
grid <- expand.grid(model=c("qwen","k26"), T=c(30,40), P=c(20,30), infl=c(1,1.5), stringsAsFactors=FALSE)
NS <- 300
out <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  g <- grid[i,]; v <- VC[[g$model]]
  r1 <- do.call(rbind, mclapply(1:NS, function(s) sim1(v, g$T, g$P, 0.10, g$infl), mc.cores=8))
  r0 <- do.call(rbind, mclapply(1:NS, function(s) sim1(v, g$T, g$P, 0.00, g$infl), mc.cores=8))
  data.frame(g, power_025=mean(r1[,"p"] < 0.025), power_05=mean(r1[,"p"] < 0.05), mean_se=mean(r1[,"se"]),
             mean_df=mean(r1[,"df"]), typeI_05=mean(r0[,"p"] < 0.05), equiv_power=mean(r0[,"equiv"]))
}))
print(out, digits=3, row.names=FALSE)
write.csv(out, "results/prereg/power_sim.csv", row.names=FALSE)
