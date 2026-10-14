# Study 2 power (drafts/study2-spec.md section 10; rule 3 of section 8). To be frozen with stamp A.
#
#   Rscript runs/s2_power.R out_dir pilot_table.csv --prior-sd S [options]   # pilot inputs -> simulation
#   Rscript runs/s2_power.R out_dir --prior-sd S                            # analytic floor only (before stamp A)
#
# --prior-sd S is the prior control-vector SD of d from Study 1's confirmatory data, the value used for the stamp A
# floor check. With a pilot table it is required and also floors the wording SD: the pilot has too few two-wording
# arguments to estimate it (2 denial arguments, DA04 and DA09), so the simulation uses the larger of the pilot
# estimate and S. The pilot estimate is reported in inputs.csv (component "aw"), and the value used as "aw_used".
#
# pilot_table.csv is runs/s2_table.py --phase pilot on the Qwen pilot (K2.6 borrows Qwen's inputs, labelled
# borrowed). The fits, fallback chain, capped Satterthwaite df and permutation test are runs/s2_confirm.R's own
# functions (sourced, not copied). Options:
#   --dose 2          Qwen pilot dose whose control rows give the inputs (rule 1's confirmatory dose)
#   --reps 500        replicates per scenario
#   --nc 14:26        control pool sizes rule 3 considers (comma list or a:b)
#   --stop            rule 3: stop at the first pool size with power >= 0.80 at beta 0.10, alpha 0.025 on the
#                     design grid (Qwen); the remaining scenarios then run at 14 and at that size only
#   --design 10x5     arguments x held-out wordings of the design grid (worst case 7x3 is always run too)
#   --n-mp 8          mental-process vectors that passed the gates (S2-G8)
#   --perm-null 200   null replicates in which the permutation p's type I error is checked
#   --cores N         parallel replicates (default detectCores() - 1)
#   --quick           skip the S2-G5, S2-G8, sensitivity and efficiency scenarios (testing only)
# Writes inputs.csv (variance components, point and upper 80% bound), floor.csv (analytic floor), power.csv (one
# row per scenario and model), efficiency.csv (SE gain per GPU hour) and timing.csv.
#
# Simulated d per cell (vector v, argument a, wording w, paraphrase p), x = 1 for emotions:
#   d = beta x + u_v (1 - x) + u_va + u_vp + u_a + x s_a + u_aw + x s_aw + u_p + x s_p + u_awp + e
# u_v, u_a, u_va at their upper 80% profile bounds (two-sided 80% profile interval, upper end); u_aw the larger of
# the two-wording arguments' estimate and --prior-sd, also 0.5x and 1x the argument SD; u_vp, u_p, u_awp, e at point estimates. The x slopes
# cannot be estimated without emotion rows: s_a = the vector:argument SD (and 2x), s_p = the vector:paraphrase
# SD (and 2x), s_aw = s_a x SD(aw) / SD(argument) (and 2x).

# Single-threaded BLAS: multithreaded OpenBLAS made one S2-H1 fit about 20x slower on a 10-core Mac (38 s vs
# 1.7 s at 3,360 rows). OpenBLAS reads the setting at load, so the script re-executes itself once with it set.
if (!interactive() && Sys.getenv("OPENBLAS_NUM_THREADS") != "1") {
  Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")
  self <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
  quit(save = "no", status = system2(file.path(R.home("bin"), "Rscript"), shQuote(c(self, commandArgs(TRUE)))))
}
suppressMessages({library(lmerTest); library(parallel)})
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
for (e in head(parse(file.path(here, "s2_confirm.R")), -1)) eval(e, globalenv())  # all but s2_confirm.R's main()
VERBOSE <- FALSE
set.seed(20261011)
RNGkind("L'Ecuyer-CMRG")

# GPU-hour accounting from section 13 (confirmatory groups, tokens per group, throughput)
GPU <- list(qwen = list(arms = 4, openings = 50, tok_group = 8680, tps = 2700, groups = 5900),
            k26 = list(arms = 2, openings = 50, tok_group = 8680, tps = 2100, groups = 2800))
TOK_ITEM <- 30; ITEMS_PER_PARA <- 12  # one more capacity paraphrase: 2 twins x 6 subjects

popt <- list(dose = 2, reps = 500, nc = 14:26, stop = FALSE, design = c(10, 5), n_mp = 8, perm_null = 200,
             cores = max(1, detectCores() - 1), quick = FALSE, prior_sd = NA)
pargs <- function(a) {
  files <- character(0); i <- 1
  num <- function(s) if (grepl(":", s)) { r <- as.integer(strsplit(s, ":")[[1]]); r[1]:r[2] } else as.numeric(strsplit(s, ",")[[1]])
  while (i <= length(a)) {
    k <- a[i]; v <- if (i < length(a)) a[i + 1] else NA
    if (k == "--dose") { popt$dose <<- as.numeric(v); i <- i + 2 }
    else if (k == "--reps") { popt$reps <<- as.integer(v); i <- i + 2 }
    else if (k == "--nc") { popt$nc <<- num(v); i <- i + 2 }
    else if (k == "--stop") { popt$stop <<- TRUE; i <- i + 1 }
    else if (k == "--design") { popt$design <<- as.integer(strsplit(v, "x")[[1]]); i <- i + 2 }
    else if (k == "--n-mp") { popt$n_mp <<- as.integer(v); i <- i + 2 }
    else if (k == "--perm-null") { popt$perm_null <<- as.integer(v); i <- i + 2 }
    else if (k == "--cores") { popt$cores <<- as.integer(v); i <- i + 2 }
    else if (k == "--quick") { popt$quick <<- TRUE; i <- i + 1 }
    else if (k == "--prior-sd") { popt$prior_sd <<- as.numeric(v); i <- i + 2 }
    else { files <- c(files, k); i <- i + 1 }
  }
  files
}

# ---- analytic floor (rule 3): SE(X) >= s_C / sqrt(nC), df nC - 1 --------------------------------------
floor_rows <- function(sC, ncs) {
  do.call(rbind, lapply(ncs, function(n) do.call(rbind, lapply(c(0.10, 0.20), function(b) do.call(rbind, lapply(c(0.025, 0.05), function(al) {
    se <- sC / sqrt(n); df <- n - 1; tc <- qt(1 - al / 2, df)
    data.frame(s_C = sC, nC = n, beta = b, alpha = al, se_floor = se,
               power_ceiling = pt(-tc, df, ncp = b / se) + pt(tc, df, ncp = b / se, lower.tail = FALSE))
  }))))))
}

# ---- variance components from the Qwen pilot control rows ----------------------------------------------
vc_from_pilot <- function(path, dose) {
  tab <<- read.csv(path, stringsAsFactors = FALSE); tab$para <<- as.character(tab$para)
  stopifnot(all(tab$phase == "pilot"))
  opt$dose[["qwen"]] <<- dose
  comp <- function(classes, label) {
    d <- cells("qwen", "S2-DEN", "d", classes)
    stopifnot(!is.null(d), nrow(d) > 0)
    f <- suppressMessages(suppressWarnings(lmer(y ~ 1 + (1 | vector) + (1 | vector:argument) + (1 | vector:para) +
           (1 | argument) + (1 | aw) + (1 | para) + (1 | aw:para), data = d, REML = TRUE,
           control = lmerControl(check.conv.singular = "ignore"))))
    vc <- as.data.frame(VarCorr(f)); sd_of <- function(g) { s <- vc$sdcor[vc$grp == g]; if (length(s)) s else 0 }
    pt <- c(vector = sd_of("vector"), va = sd_of("vector:argument"), vp = sd_of("vector:para"), arg = sd_of("argument"),
            aw = sd_of("aw"), para = sd_of("para"), awp = sd_of("aw:para"), e = sigma(f))
    up <- pt; how <- "profile"
    nm <- c(vector = "sd_(Intercept)|vector", va = "sd_(Intercept)|vector:argument", arg = "sd_(Intercept)|argument")
    ci <- tryCatch(suppressMessages(suppressWarnings(confint(f, parm = unname(nm), level = 0.80, method = "profile", oldNames = FALSE))),
                   error = function(e) NULL)
    if (is.null(ci) || any(!is.finite(ci[unname(nm), 2]))) {
      how <- "parametric bootstrap (profile failed), 200 sims"
      ci <- suppressMessages(suppressWarnings(confint(f, parm = unname(nm), level = 0.80, method = "boot", nsim = 200, oldNames = FALSE)))
    }
    up[names(nm)] <- ci[unname(nm), 2]
    data.frame(source = label, component = names(pt), point = pt, upper80 = up,
               used = ifelse(names(pt) %in% names(nm), "upper80", "point"), method = how,
               n_vectors = nlevels(droplevels(d$vector)), n_openings = length(unique(d$opening)),
               n_two_wording_args = sum(tapply(d$aw, d$argument, function(z) length(unique(z))) >= 2), row.names = NULL)
  }
  rbind(comp(PRIMARY, "primary controls"), comp("thinking", "mental-process controls"))
}

# ---- simulation ---------------------------------------------------------------------------------------
sim_cells <- function(nE, nC, A, W, P, beta, s, emo_prefix = "e", arm_tag = "") {
  vec <- c(paste0(emo_prefix, seq_len(nE)), paste0("c", seq_len(nC)))
  d <- expand.grid(vi = seq_along(vec), a = seq_len(A), w = seq_len(W), p = seq_len(P))
  d$x <- as.numeric(d$vi <= nE)
  r <- function(n, sd) rnorm(n, 0, sd)
  uv <- c(rep(0, nE), r(nC, s$vector)); ua <- r(A, s$arg); sa <- r(A, s$s_a)
  aw <- (d$a - 1) * W + d$w; uaw <- r(A * W, s$aw); saw <- r(A * W, s$s_aw)
  up <- r(P, s$para); sp <- r(P, s$s_p)
  va <- (d$vi - 1) * A + d$a; uva <- r(length(vec) * A, s$va)
  vp <- (d$vi - 1) * P + d$p; uvp <- r(length(vec) * P, s$vp)
  awp <- (aw - 1) * P + d$p; uawp <- r(A * W * P, s$awp)
  d$y <- beta * d$x + uv[d$vi] + uva[va] + uvp[vp] + ua[d$a] + d$x * sa[d$a] + uaw[aw] + d$x * saw[aw] +
         up[d$p] + d$x * sp[d$p] + uawp[awp] + r(nrow(d), s$e)
  out <- data.frame(vector = factor(paste0(arm_tag, vec[d$vi]), levels = paste0(arm_tag, vec)), argument = factor(paste0(arm_tag, "a", d$a)),
                    aw = factor(paste0(arm_tag, "a", d$a, "w", d$w)), para = factor(d$p), x = d$x, y = d$y, opening = "")
  for (j in seq_len(nE)) out[[paste0("E_", emo_prefix, j)]] <- as.numeric(d$vi == j)
  out$ctl <- 1 - out$x
  attr(out, "emo") <- paste0(emo_prefix, seq_len(nE))
  out
}

one_rep <- function(test, nE, nC, A, W, P, beta, s, perm) {
  if (test == "S2-G5") {
    a <- sim_cells(nE, nC, A, W, P, beta, s, arm_tag = "")
    b <- sim_cells(nE, nC, A, W, P, 0, s, arm_tag = "")
    # the arms' control-vector intercepts are drawn independently (correlation 0 across arms, the conservative
    # end); arguments are distinct per arm, as DA and CA are
    a$A1 <- 1; b$A1 <- 0
    d <- rbind(a, b); attr(d, "emo") <- attr(a, "emo")
    r <- fit_arm_cells(d, attr(a, "emo"), character(0), "sim", perm = perm)
  } else {
    r <- fit_h1(sim_cells(nE, nC, A, W, P, beta, s), "sim", perm = perm)
  }
  FALLBACKS <<- list()
  c(est = r$estimate, se = r$se, df = r$df, p_model = r$p_model, p_perm = r$p_perm, p_tost = r$p_tost,
    step = match(r$fit, c("maximal", "no_corr", "no_aw_slope", "no_para_slope", "no_vector_para")))
}

# one scenario: both models each replicate (K2.6 borrows Qwen's inputs), Holm across the two
scenario <- function(test, nC, A, W, beta, s, label, reps, perm_reps = reps) {
  t0 <- Sys.time()
  nE <- c(qwen = 6, k26 = 4)
  R <- mclapply(seq_len(reps), function(i) {
    sapply(names(nE), function(m) one_rep(test, nE[[m]], nC, A, W, 8, beta, s, perm = i <= perm_reps))
  }, mc.cores = popt$cores)
  R <- R[!vapply(R, inherits, logical(1), "try-error")]
  get <- function(m, k) vapply(R, function(z) z[k, m], numeric(1))
  gov <- function(m) { pm <- get(m, "p_model"); pp <- get(m, "p_perm"); if (test == "S2-G5") pm else ifelse(is.na(pp), pm, pmax(pm, pp)) }
  pq <- gov("qwen"); pk <- gov("k26")
  holm <- t(mapply(function(a, b) { al <- holm_alpha(c(a, b), c("qwen", "k26")); c(a < al[["qwen"]], b < al[["k26"]]) }, pq, pk))
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  do.call(rbind, lapply(c("qwen", "k26"), function(m) {
    p <- gov(m); pm <- get(m, "p_model"); pp <- get(m, "p_perm"); est <- get(m, "est")
    data.frame(test = test, scenario = label, model = m, inputs = if (m == "k26") "borrowed from Qwen pilot" else "Qwen pilot",
               nE = nE[[m]], nC = nC, arguments = A, wordings = W, beta = beta, reps = length(R),
               power_a025 = mean(p < 0.025), power_a05 = mean(p < 0.05), power_holm = mean(holm[, if (m == "qwen") 1 else 2]),
               p_both_holm = mean(holm[, 1] & holm[, 2]), power_model_only_a025 = mean(pm < 0.025),
               perm_reps = sum(!is.na(pp)), perm_reject_a05 = mean(pp < 0.05, na.rm = TRUE),
               tost_equiv = mean(get(m, "p_tost") < 0.05), positive_sig_a025 = mean(p < 0.025 & est > 0),
               mean_est = mean(est), mean_se = mean(get(m, "se")), mean_df = mean(get(m, "df")),
               fallback_used = mean(get(m, "step") > 1), seconds = secs)
  }))
}

main_power <- function(args) {
  out <- args[1]; files <- pargs(args[-1])
  dir.create(out, showWarnings = FALSE, recursive = TRUE)
  if (is.na(popt$prior_sd)) stop("--prior-sd is required: it is the stamp A floor's prior control SD and floors the wording SD")
  if (!length(files)) {
    fl <- floor_rows(popt$prior_sd, popt$nc); write.csv(fl, file.path(out, "floor.csv"), row.names = FALSE)
    print(format(fl, digits = 3), row.names = FALSE); return(invisible())
  }
  VC <- vc_from_pilot(files[1], popt$dose)
  aw <- VC[VC$source == "primary controls" & VC$component == "aw", ]
  VC <- rbind(VC, transform(aw, component = "aw_used", point = max(aw$point, popt$prior_sd), upper80 = NA,
                            used = "point",
                            method = sprintf("max(pilot estimate %.4g from %d two-wording arguments, --prior-sd %.4g)",
                                             aw$point, aw$n_two_wording_args, popt$prior_sd)))
  write.csv(VC, file.path(out, "inputs.csv"), row.names = FALSE)
  print(format(VC, digits = 3), row.names = FALSE)
  pick <- function(src, comp) { r <- VC[VC$source == src & VC$component == comp, ]; if (r$used == "upper80") r$upper80 else r$point }
  base <- function(src = "primary controls", aw_mode = "estimated", slope = 1) {
    s <- list(vector = pick(src, "vector"), va = pick("primary controls", "va"), vp = pick("primary controls", "vp"),
              arg = pick("primary controls", "arg"), para = pick("primary controls", "para"),
              awp = pick("primary controls", "awp"), e = pick("primary controls", "e"))
    s$aw <- switch(aw_mode, estimated = pick("primary controls", "aw_used"), half = 0.5 * s$arg, equal = s$arg)
    s$s_a <- slope * s$va; s$s_p <- slope * s$vp
    s$s_aw <- if (s$arg > 0) s$s_a * s$aw / s$arg else s$s_a
    s
  }
  fl <- floor_rows(pick("primary controls", "vector"), popt$nc)
  write.csv(fl, file.path(out, "floor.csv"), row.names = FALSE)
  P <- list(); add <- function(x) { P[[length(P) + 1]] <<- x; print(format(x[, c("test", "scenario", "model", "nC", "arguments", "wordings", "beta", "power_a025", "power_a05", "power_holm", "p_both_holm", "tost_equiv", "mean_se", "seconds")], digits = 3), row.names = FALSE) }
  grids <- list(design = popt$design, worst = c(7, 3))
  A <- popt$design[1]; W <- popt$design[2]
  # rule 3: S2-H1 power over pool sizes on the design grid at beta 0.10
  ncs <- popt$nc; chosen <- NA
  for (n in ncs) {
    r <- scenario("S2-H1", n, A, W, 0.10, base(), "base", popt$reps); add(r)
    if (is.na(chosen) && r$power_a025[r$model == "qwen"] >= 0.80) chosen <- n
    if (popt$stop && !is.na(chosen)) break
  }
  at <- unique(c(ncs[1], if (is.na(chosen)) max(ncs) else chosen))
  for (n in at) for (g in names(grids)) for (b in c(0, 0.10, 0.20)) {
    if (g == "design" && b == 0.10) next  # already run above
    add(scenario("S2-H1", n, grids[[g]][1], grids[[g]][2], b, base(), "base", popt$reps,
                 perm_reps = if (b == 0) max(popt$reps, popt$perm_null) else popt$reps))
  }
  if (!popt$quick) {
    for (g in names(grids)) for (b in c(0, 0.10, 0.20)) {
      gg <- grids[[g]]
      add(scenario("S2-H1", ncs[1], gg[1], gg[2], b, base(aw_mode = "half"), "wording SD 0.5x argument SD", popt$reps))
      add(scenario("S2-H1", ncs[1], gg[1], gg[2], b, base(aw_mode = "equal"), "wording SD 1x argument SD", popt$reps))
      add(scenario("S2-H1", ncs[1], gg[1], gg[2], b, base(slope = 2), "x slopes 2x", popt$reps))
      add(scenario("S2-G5", ncs[1], gg[1], gg[2], b, base(), "X by arm with true X(CON) 0", popt$reps))
      add(scenario("S2-G8", popt$n_mp, gg[1], gg[2], b, base("mental-process controls"), sprintf("n_mp = %d", popt$n_mp), popt$reps))
    }
  }
  PW <- do.call(rbind, P)
  PW$alpha_note <- "power_holm: Holm across the two models; power_a025/a05: one model at that alpha; p governs = max(model, permutation) except S2-G5"
  write.csv(PW, file.path(out, "power.csv"), row.names = FALSE)
  # design efficiency: SE gain per GPU hour, more controls (+2) vs one more capacity paraphrase (+12 items)
  if (!popt$quick) {
    er <- max(20, popt$reps %/% 5)  # SE is stable with fewer replicates than power
    se_at <- function(nC, Pp) { r <- mclapply(seq_len(er), function(i) one_rep("S2-H1", 6, nC, A, W, Pp, 0.10, base(), FALSE)["se"], mc.cores = popt$cores); mean(unlist(r)) }
    s0 <- se_at(ncs[1], 8); s1 <- se_at(ncs[1] + 2, 8); s2 <- se_at(ncs[1], 9)
    h_ctl <- with(GPU$qwen, 2 * arms * openings * tok_group / tps / 3600)
    h_par <- with(GPU$qwen, groups * ITEMS_PER_PARA * TOK_ITEM / tps / 3600)
    EF <- data.frame(change = c("+2 controls", "+1 capacity paraphrase"), se_base = s0, se_new = c(s1, s2),
                     gpu_hours = c(h_ctl, h_par), se_gain_per_gpu_hour = (s0 - c(s1, s2)) / c(h_ctl, h_par), reps = er)
    write.csv(EF, file.path(out, "efficiency.csv"), row.names = FALSE); print(format(EF, digits = 3), row.names = FALSE)
  }
  write.csv(data.frame(chosen_nC = chosen, reps = popt$reps, cores = popt$cores,
                       total_seconds = sum(PW$seconds[!duplicated(paste(PW$test, PW$scenario, PW$nC, PW$arguments, PW$beta))])),
            file.path(out, "timing.csv"), row.names = FALSE)
}

if (!interactive()) main_power(commandArgs(trailingOnly = TRUE))
