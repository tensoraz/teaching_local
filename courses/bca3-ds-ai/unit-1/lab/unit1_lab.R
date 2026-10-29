# Unit 1 Lab in R · Introduction to Predictive Analytics
# The same SwiftTel project as unit1_lab.py, using only base R (no packages to install).
# Run: Rscript unit1_lab.R   (or open in RStudio and press Source)
# Keep swifttel_customers.csv and swifttel_monthly.csv in the working directory.

# ---- Stage 2 · collect the data and check its types ----
df <- read.csv("swifttel_customers.csv", stringsAsFactors = TRUE)
str(df)                                   # num / int / Factor = R's data types
df$join_date <- as.Date(df$join_date)
df$city_tier <- factor(df$city_tier, levels = c("Tier-1", "Tier-2", "Tier-3"), ordered = TRUE)  # ordinal
df$satisfaction <- factor(df$satisfaction, levels = 1:5, ordered = TRUE)                         # ordinal

# ---- Stage 3 · prepare ----
colSums(is.na(df))                        # satisfaction has missing values
df$churn_flag <- as.integer(df$churned == "Yes")

# ---- Stage 4 · descriptive and diagnostic analytics ----
mean(df$churn_flag)                       # overall churn rate
tapply(df$churn_flag, df$plan_type, mean)
tapply(df$churn_flag, df$autopay, mean)
tapply(df$churn_flag, df$city_tier, mean)
aggregate(cbind(tenure_months, support_calls, monthly_charge) ~ churned, data = df, FUN = mean)

# ---- Stage 5 · predictive model (logistic regression) ----
# glm() drops the rows with a missing satisfaction score by default
model <- glm(churn_flag ~ tenure_months + monthly_charge + support_calls + plan_type + autopay
             + as.numeric(satisfaction), data = df, family = binomial)
summary(model)                            # positive estimate = more likely to churn

# ---- Stage 7 · forecast revenue with a straight-line trend ----
m <- read.csv("swifttel_monthly.csv")
fit <- lm(revenue_lakh ~ t, data = m)
coef(fit)                                 # intercept and slope: same method as Python's np.polyfit
predict(fit, data.frame(t = 25:27))       # Jan, Feb, Mar 2025
