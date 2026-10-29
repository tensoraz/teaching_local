# Unit 2 Lab in R · Data Preparation & Exploration
# The same SwiftTel cleaning pipeline as unit2_lab.py, in base R (no packages to install).
# Run: Rscript unit2_lab.R   (or Source in RStudio) with the three CSV files in the working directory.

# ---- 1 · collect: read everything as text so nothing is guessed or hidden ----
crm     <- read.csv("crm_customers.csv", colClasses = "character", encoding = "UTF-8")
usage   <- read.csv("usage_dec2024.csv")
tickets <- read.csv("support_tickets_q4_2024.csv")

# ---- 2 · clean: duplicates, labels, types ----
sum(duplicated(crm))
crm <- crm[!duplicated(crm), ]
table(crm$plan_type)
plan <- gsub("-", "", tolower(crm$plan_type))
crm$plan_type <- ifelse(plan == "prepaid", "Prepaid", "Postpaid")
g <- tolower(trimws(crm$gender))
crm$gender  <- ifelse(g %in% c("m", "male"), "Male", "Female")
crm$churned <- ifelse(tolower(crm$churned) == "yes", "Yes", "No")
crm$monthly_charge <- as.numeric(gsub("₹", "", crm$monthly_charge))      # remove the rupee sign
iso <- grepl("-", crm$join_date)                                               # two date formats
crm$join_date <- as.Date(ifelse(iso, crm$join_date, format(as.Date(crm$join_date, "%d/%m/%Y"))))
crm$age          <- as.numeric(ifelse(crm$age == "", NA, crm$age))
crm$satisfaction <- as.numeric(ifelse(crm$satisfaction == "", NA, crm$satisfaction))
str(crm)

# ---- 3 · join the sources ----
usage$customer_id <- toupper(usage$customer_id)
calls <- aggregate(ticket_id ~ customer_id, data = tickets, FUN = length)
names(calls)[2] <- "support_calls"
df <- merge(crm, usage, by = "customer_id", all.x = TRUE)                    # left join
df <- merge(df, calls, by = "customer_id", all.x = TRUE)
df$support_calls[is.na(df$support_calls)] <- 0
colSums(is.na(df))

# ---- 4 · outliers ----
q <- quantile(df$data_gb, c(.25, .75), na.rm = TRUE)
fences <- c(q[1] - 1.5 * diff(q), q[2] + 1.5 * diff(q)); fences
sum(df$data_gb < fences[1] | df$data_gb > fences[2], na.rm = TRUE)
boxplot(df$data_gb, horizontal = TRUE, main = "data_gb as received")
big <- !is.na(df$monthly_charge) & df$monthly_charge > 999
df$monthly_charge[big] <- df$monthly_charge[big] / 100
df$data_gb[!is.na(df$data_gb) & df$data_gb > 100] <- NA
df$age[!is.na(df$age) & (df$age < 18 | df$age > 100)] <- NA

# ---- 5 · missing values ----
tapply(is.na(df$age), df$plan_type, mean)                                     # MAR: depends on plan
tapply(df$churned == "Yes", is.na(df$satisfaction), mean)                     # MNAR hint
df$age_was_missing <- as.integer(is.na(df$age))
med_by_plan <- tapply(df$age, df$plan_type, median, na.rm = TRUE)
df$age[is.na(df$age)] <- med_by_plan[df$plan_type[is.na(df$age)]]
df$satisfaction[is.na(df$satisfaction)] <- median(df$satisfaction, na.rm = TRUE)
for (col in c("data_gb", "voice_minutes", "recharge_count"))
  df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)

# ---- 6 · transformation ----
df$tenure_months <- pmin(pmax(floor(as.numeric(as.Date("2025-01-01") - df$join_date) / 30.4), 1), 72)
df$churn_flag    <- as.integer(df$churned == "Yes")
df$charge_minmax <- (df$monthly_charge - min(df$monthly_charge)) / diff(range(df$monthly_charge))
df$charge_z      <- as.numeric(scale(df$monthly_charge))
df$log_voice     <- log1p(df$voice_minutes)
df$tenure_band   <- cut(df$tenure_months, c(0, 6, 12, 24, 48, 72))
df$city_tier     <- factor(df$city_tier, levels = c("Tier-1", "Tier-2", "Tier-3"), ordered = TRUE)

# ---- 7 · feature selection ----
num <- c("tenure_months", "monthly_charge", "data_gb", "voice_minutes", "support_calls", "age", "satisfaction")
sort(abs(cor(df[num], df$churn_flag)[, 1]), decreasing = TRUE)
cor(df$data_gb, df$monthly_charge)                                            # redundancy
chisq.test(table(df$plan_type, df$churned), correct = FALSE)
chisq.test(table(df$gender, df$churned), correct = FALSE)

# ---- 8 · EDA ----
summary(df[num])
hist(df$voice_minutes, breaks = 40, main = "voice minutes")
boxplot(tenure_months ~ churned, data = df, main = "tenure by churn")
plot(df$monthly_charge, df$data_gb, pch = 16, cex = .5)
heatmap(cor(cbind(df[num], churn = df$churn_flag)), symm = TRUE)
write.csv(df, "swifttel_prepared_R.csv", row.names = FALSE)
