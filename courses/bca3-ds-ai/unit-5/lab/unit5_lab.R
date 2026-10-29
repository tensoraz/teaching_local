# Unit 5 Lab in R · Model Deployment & Applications
# Interpretation, saving and batch scoring, drift (PSI), a fairness audit and K-means, all in base R.
# Run: Rscript unit5_lab.R with the CSV files from this unit in the working directory.

set.seed(42)
credit <- read.csv("credit_applications.csv")
credit$self_employed <- as.integer(credit$employment == "Self-employed")
idx <- sample(nrow(credit), 0.7 * nrow(credit))
train <- credit[idx, ]; test <- credit[-idx, ]
f <- defaulted ~ age + years_employed + income_lakh + loan_amount_lakh + emi_to_income + credit_score + past_defaults + self_employed

# ---- 1 · model and global interpretation ----
model <- glm(f, data = train, family = binomial)
summary(model)
exp(coef(model))                                     # odds ratios per +1 unit
exp(coef(model)["credit_score"] * 50)                # odds ratio for +50 credit points
p <- predict(model, test, type = "response")
auc <- function(p, y) { pos <- p[y == 1]; neg <- p[y == 0]; mean(outer(pos, neg, ">") + 0.5 * outer(pos, neg, "==")) }
base_auc <- auc(p, test$defaulted); base_auc

# permutation importance: shuffle one column, measure the AUC drop
perm <- sapply(c("credit_score", "emi_to_income", "past_defaults", "self_employed", "income_lakh", "age"), function(col) {
  shuffled <- test; shuffled[[col]] <- sample(shuffled[[col]])
  base_auc - auc(predict(model, shuffled, type = "response"), test$defaulted)
})
sort(perm, decreasing = TRUE)

# ---- 2 · local explanation: reason codes for one applicant ----
THRESHOLD <- 0.25
x <- test[which(p > THRESHOLD)[1], ]
vars <- c("age", "years_employed", "income_lakh", "loan_amount_lakh", "emi_to_income", "credit_score", "past_defaults", "self_employed")
contrib <- coef(model)[vars] * (unlist(x[vars]) - colMeans(train[vars]))
sort(contrib, decreasing = TRUE)                     # positive = pushes towards default

# ---- 3 · deploy: save, reload, batch-score ----
saveRDS(model, "credit_model.rds")
loaded <- readRDS("credit_model.rds")
new <- read.csv("new_applications.csv"); new$self_employed <- as.integer(new$employment == "Self-employed")
new$p_default <- round(predict(loaded, new, type = "response"), 3)
new$decision  <- ifelse(new$p_default < THRESHOLD, "approve", "refer to officer")
write.csv(new[c("application_id", "p_default", "decision")], "scored_applications_R.csv", row.names = FALSE)
table(new$decision)

# ---- 4 · monitor: PSI ----
this_month <- read.csv("credit_scores_this_month.csv")$credit_score
edges <- quantile(train$credit_score, seq(0, 1, 0.1)); edges[1] <- 300; edges[11] <- 901
e <- as.numeric(table(cut(train$credit_score, edges, right = FALSE))) / nrow(train)
a <- as.numeric(table(cut(this_month, edges, right = FALSE))) / length(this_month)
psi <- sum((a - e) * log(a / e)); psi                # > 0.25 = significant drift

# ---- 5 · fairness audit ----
test$approved <- as.integer(p < THRESHOLD)
aggregate(cbind(approved, defaulted) ~ region, data = test, FUN = mean)
aggregate(approved ~ region, data = test[test$defaulted == 0, ], FUN = mean)       # approval among good payers
rates <- tapply(test$approved, test$region, mean); min(rates) / max(rates)        # disparate impact ratio

# ---- 6 · healthcare: screening threshold ----
clinic <- read.csv("clinic_screening.csv"); clinic$low_activity <- as.integer(clinic$activity_level == "Low")
h <- glm(diabetic ~ age + bmi + fasting_glucose + systolic_bp + family_history + low_activity, data = clinic, family = binomial)
hp <- predict(h, type = "response")
sapply(c(0.5, 0.2, 0.1), function(t) c(threshold = t, recall = mean(hp[clinic$diabetic == 1] >= t), referred = sum(hp >= t)))

# ---- 7 · marketing: K-means segments (Lab Experiment 9) ----
sw <- read.csv("swifttel_prepared.csv")
Z <- scale(sw[c("data_gb", "log_voice", "monthly_charge", "tenure_months")])
inertia <- sapply(1:8, function(k) kmeans(Z, k, nstart = 10)$tot.withinss)
plot(1:8, inertia, type = "b", main = "Elbow method", xlab = "k")
km <- kmeans(Z, 4, nstart = 10)
sw$segment <- km$cluster
aggregate(cbind(data_gb, voice_minutes, monthly_charge, tenure_months, churn_flag) ~ segment, data = sw, FUN = mean)
