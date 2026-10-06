# Exploratory: does the higher dose beat the lower one? Per task (40), the emotions-minus-controls contrast on
# self_resid at the higher dose minus at the lower dose, paired t-test across tasks. (A mixed model with an x:hi term
# and no random slope for it is anticonservative here, so it is not used.)
#
#   Rscript runs/confirm_dose_test.R
CTRL <- c("red", "ocean", "random", "arithmetic")
cfg <- list(qwen = list(f = "qwen", layer = 40, doses = c(1, 2), emo = c("joy", "sadness", "anger", "fear", "euphoria", "surprise")),
            k26 = list(f = "k26", layer = 20, doses = c(2, 4), emo = c("joy", "fear", "euphoria", "surprise")))
for (m in names(cfg)) {
  c <- cfg[[m]]
  t <- read.csv(sprintf("results/confirm/tables/%s.csv", c$f), stringsAsFactors = FALSE)
  t <- t[t$measure == "self_resid" & t$site == "answer" & t$layer == c$layer & t$vec %in% c(c$emo, CTRL), ]
  for (a in c("D0", "D1.4")) {
    d <- t[t$arm == a & t$dose %in% c$doses, ]
    d$x <- as.numeric(d$vec %in% c$emo); d$hi <- as.numeric(d$dose == max(c$doses))
    # per-task paired test of the emotion-minus-control contrast, high dose minus low dose
    g <- aggregate(value ~ task + x + hi, d, mean)
    con <- function(h) { s <- g[g$hi == h, ]; tapply(s$value[s$x == 1], s$task[s$x == 1], identity) - tapply(s$value[s$x == 0], s$task[s$x == 0], identity) }
    diff <- con(1) - con(0)
    tt <- t.test(diff)
    cat(sprintf("%-4s %-5s low %.3f high %.3f | task-paired diff %+.3f [%.3f, %.3f] p=%.2g, %d/40 tasks up\n",
                m, a, mean(con(0)), mean(con(1)), mean(diff), tt$conf.int[1], tt$conf.int[2], tt$p.value, sum(diff > 0)))
  }
}
