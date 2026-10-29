# Unit 3 Lab in R · Predictive Modeling Techniques
# Base R for regression, logistic regression and Naive Bayes by counting; packages for tree and KNN.
# Optional packages: install.packages(c("rpart", "rpart.plot", "class"))
# Run: Rscript unit3_lab.R with used_cars.csv and swifttel_customers.csv in the working directory.

set.seed(42)

# ---- Regression: split the car data 70 / 30 ----
cars <- read.csv("used_cars.csv", stringsAsFactors = TRUE)
cars$brand_tier <- relevel(cars$brand_tier, ref = "Budget")      # baseline category
cars$fuel       <- relevel(cars$fuel, ref = "Petrol")
idx   <- sample(nrow(cars), 0.7 * nrow(cars))
train <- cars[idx, ]; test <- cars[-idx, ]

rmse <- function(a, p) sqrt(mean((a - p)^2))
r2   <- function(a, p) 1 - sum((a - p)^2) / sum((a - mean(a))^2)

simple <- lm(price_lakh ~ age_years, data = train)
summary(simple)                                                   # slope, intercept, R-squared
p <- predict(simple, test); c(R2 = r2(test$price_lakh, p), RMSE = rmse(test$price_lakh, p))

multi <- lm(price_lakh ~ age_years + I(km_driven / 1000) + owners + brand_tier + fuel + transmission, data = train)
summary(multi)                                                    # coefficients, adjusted R-squared, p-values
p <- predict(multi, test); c(R2 = r2(test$price_lakh, p), RMSE = rmse(test$price_lakh, p))
plot(fitted(multi), resid(multi), main = "Residuals vs fitted"); abline(h = 0, col = "orange")

# ---- Classification: SwiftTel churn ----
df <- read.csv("swifttel_customers.csv")
df$churn   <- as.integer(df$churned == "Yes")
df$prepaid <- as.integer(df$plan_type == "Prepaid")
df$autopay <- as.integer(df$autopay == "Yes")
idx <- sample(nrow(df), 0.7 * nrow(df))
tr <- df[idx, ]; te <- df[-idx, ]
med <- median(tr$satisfaction, na.rm = TRUE)                      # learned from training data only
tr$satisfaction[is.na(tr$satisfaction)] <- med
te$satisfaction[is.na(te$satisfaction)] <- med
f <- churn ~ tenure_months + monthly_charge + support_calls + prepaid + autopay + satisfaction

logit <- glm(f, data = tr, family = binomial)
summary(logit)
exp(coef(logit))                                                  # odds ratios
p <- predict(logit, te, type = "response")
table(predicted = p > 0.5, actual = te$churn)
mean((p > 0.5) == te$churn)                                       # accuracy

if (requireNamespace("rpart", quietly = TRUE)) {
  tree <- rpart::rpart(f, data = tr, method = "class", control = rpart::rpart.control(maxdepth = 3))
  print(tree)
  if (requireNamespace("rpart.plot", quietly = TRUE)) rpart.plot::rpart.plot(tree)
}
if (requireNamespace("class", quietly = TRUE)) {
  cols <- c("tenure_months", "monthly_charge", "support_calls", "prepaid", "autopay", "satisfaction")
  m <- colMeans(tr[cols]); s <- apply(tr[cols], 2, sd)            # scale with TRAINING mean and sd
  knn_pred <- class::knn(scale(tr[cols], m, s), scale(te[cols], m, s), cl = tr$churn, k = 15)
  mean(knn_pred == te$churn)
}

# ---- Naive Bayes by counting (the worked example) ----
P <- function(cond, cls) mean(cond[df$churn == cls])
evidence <- with(df, list(support_calls >= 3, plan_type == "Prepaid", autopay == 0, tenure_months <= 12))
num_c <- mean(df$churn == 1) * prod(sapply(evidence, P, cls = 1))
num_s <- mean(df$churn == 0) * prod(sapply(evidence, P, cls = 0))
num_c / (num_c + num_s)                                           # posterior P(churn | evidence)
