# Unit 4 Lab in R · Advanced Modeling & Validation
# k-fold cross-validation and metrics in base R; ensembles with optional packages.
# Optional: install.packages(c("rpart", "randomForest", "gbm", "pROC"))
# Run: Rscript unit4_lab.R with swifttel_customers.csv in the working directory.

set.seed(42)
df <- read.csv("swifttel_customers.csv")
df$churn   <- as.integer(df$churned == "Yes")
df$prepaid <- as.integer(df$plan_type == "Prepaid")
df$autopay <- as.integer(df$autopay == "Yes")
df$tier3   <- as.integer(df$city_tier == "Tier-3")
df$satisfaction[is.na(df$satisfaction)] <- median(df$satisfaction, na.rm = TRUE)
f <- churn ~ tenure_months + monthly_charge + support_calls + prepaid + autopay + tier3 + satisfaction + age + data_gb

# ---- locked test set (stratified by hand: sample 30% of each class) ----
test_idx <- c(sample(which(df$churn == 1), round(.3 * sum(df$churn))), sample(which(df$churn == 0), round(.3 * sum(df$churn == 0))))
train <- df[-test_idx, ]; test <- df[test_idx, ]

# ---- AUC without packages: probability a random churner scores above a random stayer ----
auc <- function(p, y) { pos <- p[y == 1]; neg <- p[y == 0]; mean(outer(pos, neg, ">") + 0.5 * outer(pos, neg, "==")) }

# ---- 5-fold cross-validation of logistic regression ----
folds <- sample(rep(1:5, length.out = nrow(train)))
cv <- sapply(1:5, function(k) {
  fit <- glm(f, data = train[folds != k, ], family = binomial)
  auc(predict(fit, train[folds == k, ], type = "response"), train$churn[folds == k])
})
round(cv, 3); c(mean = mean(cv), sd = sd(cv))

# ---- final logistic model, metrics at two thresholds ----
logit <- glm(f, data = train, family = binomial)
p <- predict(logit, test, type = "response")
auc(p, test$churn)
metrics <- function(thr) {
  pred <- as.integer(p >= thr); y <- test$churn
  tp <- sum(pred == 1 & y == 1); fp <- sum(pred == 1 & y == 0); fn <- sum(pred == 0 & y == 1); tn <- sum(pred == 0 & y == 0)
  prec <- tp / (tp + fp); rec <- tp / (tp + fn)
  c(threshold = thr, TP = tp, FP = fp, FN = fn, TN = tn, accuracy = (tp + tn) / length(y),
    precision = prec, recall = rec, specificity = tn / (tn + fp), F1 = 2 * prec * rec / (prec + rec))
}
rbind(metrics(0.5), metrics(0.25))

# ---- tree depth: under- and overfitting ----
if (requireNamespace("rpart", quietly = TRUE)) {
  for (d in c(1, 2, 4, 8, 15)) {
    t <- rpart::rpart(f, data = train, method = "class", control = rpart::rpart.control(maxdepth = d, cp = 0, minsplit = 2))
    cat("depth", d, " train AUC", round(auc(predict(t, train)[, 2], train$churn), 3),
        " test AUC", round(auc(predict(t, test)[, 2], test$churn), 3), "\n")
  }
}

# ---- ensembles ----
if (requireNamespace("randomForest", quietly = TRUE)) {
  rf <- randomForest::randomForest(factor(churn) ~ tenure_months + monthly_charge + support_calls + prepaid + autopay + tier3 +
                                     satisfaction + age + data_gb, data = train, ntree = 100, nodesize = 20)
  print(rf)                                              # includes the out-of-bag error estimate
  auc(predict(rf, test, type = "prob")[, 2], test$churn)
  randomForest::varImpPlot(rf)
}
if (requireNamespace("gbm", quietly = TRUE)) {
  gb <- gbm::gbm(f, data = train, distribution = "bernoulli", n.trees = 150, interaction.depth = 2, shrinkage = 0.05)
  auc(predict(gb, test, n.trees = 150, type = "response"), test$churn)
}
if (requireNamespace("pROC", quietly = TRUE)) plot(pROC::roc(test$churn, p), main = "ROC: logistic regression")
