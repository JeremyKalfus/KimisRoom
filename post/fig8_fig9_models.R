# Mixed models behind figures 8 and 9 of the LessWrong post, for the cells no file in results/confirm/analysis holds.
#
#   Rscript post/fig8_fig9_models.R post/fig8_fig9_models.csv     # from the repository root
#
# The model code is the registered one: runs/confirm.R's fit_test, with its fallback order, through
# runs/confirm_explore.R's selections (registered layer, primary dose, vector on the answer). Exploratory.
#   figure 8, left   "id_self - id_user" on the two Qwen fine-tunes, intercept-only on task x paraphrase cells,
#                    as confirm_explore.R does for the base models (their rows are in explore.csv and, under
#                    the argument, hypotheses.csv H4q/H4k)
#   figure 8, right  Assistant vs User on who_felt_it (measure who_user), intercept-only, all six conditions
#   figure 9         emotions minus controls on belief, the preregistered belief measure, all six conditions;
#                    the four base-model rows are the "emotions - controls on belief" rows of explore.csv
# Tables, results/confirm/tables/<name>.csv or <name>.csv.gz: qwen, k26, qwen-D2, qwen-D2c, and who_user
# (written by post/who_user_table.py). Needs R >= 4.3 with lme4 and lmerTest.

suppressMessages(library(lmerTest))
for (e in head(parse("runs/confirm_explore.R"), -1)) eval(e, globalenv())  # everything but its main() call

tbl <- function(name) {
  p <- file.path("results/confirm/tables", paste0(name, ".csv"))
  read.csv(if (file.exists(p)) p else paste0(p, ".gz"), stringsAsFactors = FALSE)
}
CONDS <- list(c("qwen", "D0"), c("qwen", "D1.4"), c("qwen-D2c", "D0"), c("qwen-D2", "D0"), c("k26", "D0"), c("k26", "D1.4"))
emo_of <- function(model) CFG[[sub("-.*", "", model)]]$emo
X <- list()
add <- function(figure, r) X[[length(X) + 1]] <<- cbind(figure = figure, r)

tab <<- rbind(tbl("qwen-D2"), tbl("qwen-D2c"))
for (m in c("qwen-D2c", "qwen-D2"))
  add("8 left", row("identification", m, "D0", "id_self - id_user", margin_cells(m, "D0", "id_self", "id_user"),
                    coef = "(Intercept)"))

tab <<- tbl("who_user")
for (c in CONDS)
  add("8 right", row("who_felt_it", c[1], c[2], "Assistant - User", mean_cells(c[1], c[2], "who_user", emo_of(c[1])),
                     coef = "(Intercept)"))

tab <<- do.call(rbind, lapply(c("qwen", "k26", "qwen-D2", "qwen-D2c"), function(n) subset(tbl(n), measure == "belief")))
for (c in CONDS)
  add("9", row("belief", c[1], c[2], "emotions - controls on belief",
               contrast(c[1], c[2], emo_of(c[1]), CTRL, "belief")))

X <- do.call(rbind, X)
write.csv(X, commandArgs(trailingOnly = TRUE)[1], row.names = FALSE)
print(format(X[, c("figure", "model", "arm", "estimate", "se", "df", "p", "ci95_lo", "ci95_hi", "fit", "singular")],
             digits = 4), row.names = FALSE)
