import csv
import math
import os
import numpy as np


INPUT_FILE = "data/records.csv"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "part1_results.txt")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

with open(INPUT_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

print("Loaded records:", len(rows))

if len(rows) != 800:
    raise ValueError(f"Expected 800 rows, found {len(rows)}")

def column(name):
    return np.array(
        [float(row[name]) for row in rows],
        dtype=float
    )


#Predictor Variables
X1 = column("months_registered")
X2 = column("missed_visit_count_12m")
X3 = column("copay_burden_ratio")
X4 = column("avg_bill_inr_thousands")
X5 = column("adherence_score")
X6 = column("chronic_condition_flag")
X7 = column("insured_flag")

#Target Variable
y = column("care_plan_cost_inr_thousands")

n = len(y)
res = {}

# Estimation with 95% confidence interval
mean_value = np.mean(y)
sd_value = np.std(y, ddof=1)
se_value = sd_value / math.sqrt(n)

ci95_low = mean_value - 1.96 * se_value
ci95_high = mean_value + 1.96 * se_value

res["n"] = n
res["mean_value"] = mean_value
res["sd_value"] = sd_value
res["se_value"] = se_value
res["ci95_low"] = ci95_low
res["ci95_high"] = ci95_high

# Hypothesis Testing
planning_claim = 8.0
alpha = 0.05
critical_value = 1.96
z_stat = (mean_value - planning_claim) / se_value
reject_h0 = "yes" if abs(z_stat) > critical_value else "no"
res["z_stat"] = z_stat
res["reject_h0"] = reject_h0

#Required Sample Size
margin_error = 0.5
n_required = math.ceil(
    (critical_value * sd_value / margin_error) ** 2
)
res["n_required"] = n_required

#Closed form simple linear regression

def simple_regression(x, target):
    count = len(target)
    x_mean = np.mean(x)
    y_mean = np.mean(target)
    covariance = np.sum((x - x_mean) * (target - y_mean)) / (count - 1)
    variance = np.sum(
        (x - x_mean) ** 2
    ) / (count - 1)
    slope = covariance / variance
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * x
    residuals = target - fitted
    tss = np.sum((target - y_mean) ** 2)
    ess = np.sum((fitted - y_mean) ** 2)
    rss = np.sum(residuals ** 2)
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / count)
    mae = np.mean(np.abs(residuals))
    return {
        "slope": slope,
        "intercept":intercept,
        "fitted" : fitted,
        "residuals": residuals,
        "tss":tss,
        "ess":ess,
        "rss":rss,
        "r2":r2,
        "rmse":rmse,
        "mae":mae
    }


#Simple Regression: Closed Form
reg = simple_regression(X4, y)
reg_slope = reg["slope"]
reg_intercept = reg["intercept"]
fitted = reg["fitted"]
residuals = reg["residuals"]
tss = reg["tss"]
ess = reg["ess"]
rss = reg["rss"]

tss_equals_ess_plus_rss = (
    "yes"
    if abs(tss - ess - rss) < 1e-6 * tss else "no"
)

r2 = reg["r2"]
rmse = reg["rmse"]
mae = reg["mae"]

res["reg_slope"] = reg_slope
res["reg_intercept"] = reg_intercept
res["tss"] = tss
res["ess"] = ess
res["rss"] = rss
res["tss_equals_ess_plus_rss"] = tss_equals_ess_plus_rss
res["r2"] = r2
res["rmse"] = rmse
res["mae"] = mae

#Gradient Descent
x_mean = np.mean(X4)
x_sd = np.std(X4, ddof = 1)
z = (X4 - x_mean) / x_sd
learning_rate = 0.1
iterations = 500
m = 0.0
c = 0.0

def gd_cost(slope, intercept):
    predictions = slope * z + intercept
    errors = y - predictions
    return np.mean(errors ** 2)

#Initial cost
gd_J0 = gd_cost(m,c)
gd_J1 = None
gd_J10 = None
gd_J100 = None
gd_J500 = None

for iteration in range(1, iterations + 1):
    predictions = m * z + c
    errors = y - predictions

    dm = -(2.0 / n) * np.sum(z * errors)
    dc = -(2.0 / n) * np.sum(errors)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

    if iteration in (1, 10 , 100, 500):
        cost = gd_cost(m, c)
        if(iteration == 1):
            gd_J1 = cost
        elif iteration == 10:
            gd_J10 = cost
        elif iteration == 100:
            gd_J100 = cost
        elif iteration == 500:
            gd_J500 = cost
gd_slope = m / x_sd
gd_intercept = c - m * x_mean / x_sd

gd_slope_gap = abs(gd_slope - reg_slope)

res["gd_J0"] = gd_J0
res["gd_J1"] = gd_J1
res["gd_J10"] = gd_J10
res["gd_J100"] = gd_J100
res["gd_J500"] = gd_J500
res["gd_slope"] = gd_slope
res["gd_intercept"] = gd_intercept
res["gd_slope_gap"] = gd_slope_gap

# Funnel Check and LOG-LOG regression

raw_order = np.argsort(fitted, kind='stable')
raw_lower_indices = raw_order[:400]
raw_upper_indices = raw_order[400:]

resid_sd_lower_raw = np.std(
    residuals[raw_lower_indices], ddof=1
)

resid_sd_upper_raw = np.std(
    residuals[raw_upper_indices], ddof=1
)

resid_sd_ratio_raw = (
    resid_sd_upper_raw / resid_sd_lower_raw
)

#Log-Log Regression
log_x4 = np.log(X4)
log_y = np.log(y)

log_reg = simple_regression(log_x4, log_y)

loglog_intercept = log_reg["intercept"]
loglog_slope = log_reg["slope"]
loglog_fitted = log_reg["fitted"]
loglog_residuals = log_reg["residuals"]

loglog_r2 = log_reg["r2"]
loglog_rmse = log_reg["rmse"]

log_order = np.argsort(loglog_fitted, kind="stable")

log_lower_indices = log_order[:400]
log_upper_indices = log_order[400:]

resid_sd_lower_log = np.std(
    loglog_residuals[log_lower_indices], ddof=1
)

resid_sd_upper_log = np.std(
    loglog_residuals[log_upper_indices], ddof = 1
)

resid_sd_ratio_log = (
    resid_sd_upper_log / resid_sd_lower_log
)

loglog_adj_r2 = 1.0 - (
    (1.0 - loglog_r2) * (n - 1) / (n -2)
)

res["resid_sd_lower_raw"] = resid_sd_lower_raw
res["resid_sd_upper_raw"] = resid_sd_upper_raw
res["resid_sd_ratio_raw"] = resid_sd_ratio_raw

res["loglog_intercept"] = loglog_intercept
res["loglog_slope"] = loglog_slope
res["loglog_r2"] = loglog_r2
res["loglog_rmse"] = loglog_rmse

res["resid_sd_lower_log"] = resid_sd_lower_log
res["resid_sd_upper_log"] = resid_sd_upper_log
res["resid_sd_ratio_log"] = resid_sd_ratio_log

res["loglog_adj_r2"] = loglog_adj_r2

# Multiple Regression

design_matrix = np.column_stack([
    np.ones(n),
    np.log(X4),
    X1,
    X2,
    X3,
    X5,
    X6,
    X7
])

target_log = np.log(y)

xtx = design_matrix.T @ design_matrix
xty = design_matrix.T @ target_log

beta = np.linalg.solve(xtx, xty)

multiple_fitted = design_matrix @ beta
multiple_residuals = target_log - multiple_fitted

mr_tss = np.sum(
    (target_log - np.mean(target_log)) ** 2
)

mr_rss = np.sum(multiple_residuals ** 2)

mr_r2 = 1.0 - mr_rss / mr_tss

p = 7

mr_adj_r2 = 1.0 - (
    (1.0 - mr_r2) * (n - 1) / (n - p - 1)
)

for i in range(8):
    res[f"mr_beta_{i}"] = beta[i]

res["mr_r2"] = mr_r2
res["mr_adj_r2"] = mr_adj_r2


os.makedirs(OUTPUT_DIR, exist_ok = True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for key, value in res.items():
        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")
        else:
            file.write(f"{key} = {value}\n")


print("=" * 55)
print("PHARMEASY HEALTHCARE - PART 1 RESULTS")
print("=" * 55)\

for key, value in res.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key:30s} = {value:.4f}")
    else:
        print(f"{key:30s} = {value}")
print("=" * 55)
print(f"Results saved to : {OUTPUT_FILE}")
print("=" * 55)
