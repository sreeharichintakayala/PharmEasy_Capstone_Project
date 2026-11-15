
"""
PharmEasy Healthcare Decision Intelligence
Part 2: Classification and Evaluation

Run:
    python part2.py

Input:
    data/records.csv

Outputs:
    outputs/part2_results.txt
    outputs/part2_thresholds.csv
    outputs/part2_knn_cv.csv

Logistic regression and KNN are implemented from scratch.
"""

import csv
import math
import os
import numpy as np


# ==================================================
# 1. LOADING DATA
# ==================================================

INPUT_FILE = "data/records.csv"
OUTPUT_DIR = "outputs"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR, "part2_results.txt"
)

THRESHOLDS_FILE = os.path.join(
    OUTPUT_DIR, "part2_thresholds.csv"
)

KNN_CV_FILE = os.path.join(
    OUTPUT_DIR, "part2_knn_cv.csv"
)

with open(INPUT_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

if len(rows) != 800:
    raise ValueError(
        f"Expected 800 records, found {len(rows)}"
    )


def column(name):
    return np.array(
        [float(row[name]) for row in rows],
        dtype=float
    )


# Seven predictors in exact order
X_raw = np.column_stack([
    column("months_registered"),
    column("missed_visit_count_12m"),
    column("copay_burden_ratio"),
    column("avg_bill_inr_thousands"),
    column("adherence_score"),
    column("chronic_condition_flag"),
    column("insured_flag")
])

y = column("no_show")

N = len(y)
results = {}


# ==================================================
# 2.1 STRATIFIED 70/30 SPLIT
# ==================================================

# Preserve original file order within each class.
negative_indices = np.where(y == 0)[0]
positive_indices = np.where(y == 1)[0]

# Integer arithmetic as specified.
negative_train_count = (
    7 * len(negative_indices)
) // 10

positive_train_count = (
    7 * len(positive_indices)
) // 10

negative_train = negative_indices[:negative_train_count]
negative_test = negative_indices[negative_train_count:]

positive_train = positive_indices[:positive_train_count]
positive_test = positive_indices[positive_train_count:]

# Combine and restore original file order.
train_indices = np.sort(
    np.concatenate([
        negative_train,
        positive_train
    ])
)

test_indices = np.sort(
    np.concatenate([
        negative_test,
        positive_test
    ])
)

X_train_raw = X_raw[train_indices]
X_test_raw = X_raw[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

train_rows = len(y_train)
test_rows = len(y_test)

train_positives = int(np.sum(y_train))
test_positives = int(np.sum(y_test))

results["train_rows"] = train_rows
results["train_positives"] = train_positives
results["test_rows"] = test_rows
results["test_positives"] = test_positives

# Validate assignment checkpoint.
if (
    train_rows != 560
    or train_positives != 119
    or test_rows != 240
    or test_positives != 51
):
    raise ValueError(
        "Stratified split does not match the assignment checkpoint."
    )


# ==================================================
# STANDARDISATION USING TRAINING DATA ONLY
# ==================================================

train_mean = np.mean(X_train_raw, axis=0)
train_sd = np.std(X_train_raw, axis=0, ddof=1)

if np.any(train_sd == 0):
    raise ValueError("A predictor has zero training SD.")

X_train = (X_train_raw - train_mean) / train_sd
X_test = (X_test_raw - train_mean) / train_sd

# Do not calculate new statistics using X_test.


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def sigmoid(z):
    """Numerically stable enough for this dataset."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(actual, probabilities):
    """Mean binary cross-entropy."""
    eps = 1e-15

    probabilities = np.clip(
        probabilities, eps, 1.0 - eps
    )

    loss = -np.mean(
        actual * np.log(probabilities)
        + (1.0 - actual) * np.log(1.0 - probabilities)
    )

    return float(loss)


def confusion_metrics(actual, predicted):
    """Calculate confusion cells and classification metrics."""

    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    def divide(a, b):
        return a / b if b != 0 else 0.0

    accuracy = divide(tp + tn, tp + fp + tn + fn)
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)

    f1 = divide(
        2 * precision * recall,
        precision + recall
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }


# ==================================================
# LOGISTIC REGRESSION FROM SCRATCH
# ==================================================

def fit_logistic(X, target, learning_rate, updates):


    sample_count, feature_count = X.shape

    weights = np.zeros(feature_count, dtype=float)
    bias = 0.0

    def calculate_loss():
        probabilities = sigmoid(X @ weights + bias)
        return binary_cross_entropy(target, probabilities)

    losses = {0: calculate_loss()}

    for epoch in range(1, updates + 1):

        # Forward pass
        z = X @ weights + bias
        probabilities = sigmoid(z)

        # Error
        error = probabilities - target

        # Gradients
        dw = (X.T @ error) / sample_count
        db = np.sum(error) / sample_count

        # Update
        weights -= learning_rate * dw
        bias -= learning_rate * db

        if epoch in (1, 10, 100, 200, 1000):
            losses[epoch] = calculate_loss()

    return weights, bias, losses


# ==================================================
# 2.2 LOGISTIC REGRESSION: 1000 UPDATES
# ==================================================

weights, bias, losses = fit_logistic(
    X_train,
    y_train,
    learning_rate=0.1,
    updates=1000
)

results["loss_epoch_0"] = losses[0]
results["loss_epoch_1"] = losses[1]
results["loss_epoch_10"] = losses[10]
results["loss_epoch_100"] = losses[100]
results["loss_epoch_1000"] = losses[1000]

for i in range(7):
    results[f"weight_X{i + 1}"] = weights[i]

results["bias"] = bias

# Test evaluation only after model fitting.
test_probabilities = sigmoid(
    X_test @ weights + bias
)

test_bce = binary_cross_entropy(
    y_test,
    test_probabilities
)

results["test_bce"] = test_bce


# ==================================================
# 2.3 LEARNING RATE COMPARISON
# ==================================================

learning_rates = [0.01, 0.1, 1.0]

for eta in learning_rates:

    _, _, eta_losses = fit_logistic(
        X_train,
        y_train,
        learning_rate=eta,
        updates=200
    )

    key = f"loss_epoch_200_eta_{eta}"

    results[key] = eta_losses[200]


# ==================================================
# 2.4 ODDS RATIOS
# ==================================================

for i in range(7):
    results[f"odds_ratio_X{i + 1}"] = math.exp(weights[i])


# ==================================================
# 2.5 TEST EVALUATION AT 0.5 AND 0.3
# ==================================================

for tau in (0.5, 0.3):

    predictions = (
        test_probabilities >= tau
    ).astype(int)

    metrics = confusion_metrics(
        y_test,
        predictions
    )

    prefix = f"tau_{tau}"

    for key, value in metrics.items():
        results[f"{prefix}_{key}"] = value


# Blind model: predict zero for every test record.
blind_predictions = np.zeros(
    len(y_test), dtype=int
)

blind_metrics = confusion_metrics(
    y_test,
    blind_predictions
)

results["blind_accuracy"] = blind_metrics["accuracy"]


# ==================================================
# 2.6 ROC THRESHOLD SWEEP
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(
    THRESHOLDS_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "tau", "tp", "fp", "tn", "fn", "tpr", "fpr"
    ])

    for tau in np.arange(0.1, 1.0, 0.1):

        tau = round(float(tau), 1)

        predictions = (
            test_probabilities >= tau
        ).astype(int)

        metrics = confusion_metrics(
            y_test,
            predictions
        )

        tpr = metrics["recall"]
        fpr = (
            metrics["fp"] /
            (metrics["fp"] + metrics["tn"])
            if (metrics["fp"] + metrics["tn"]) > 0
            else 0.0
        )

        writer.writerow([
            f"{tau:.4f}",
            metrics["tp"],
            metrics["fp"],
            metrics["tn"],
            metrics["fn"],
            f"{tpr:.4f}",
            f"{fpr:.4f}"
        ])


# ==================================================
# 2.7 KNN WITH 5-FOLD CROSS-VALIDATION
# ==================================================

def knn_predict(
    reference_X,
    reference_y,
    query_X,
    k
):
    """
    Euclidean KNN.

    Stable sorting preserves reference-row order
    when distances are tied.
    """

    # Calculate Euclidean distances.
    differences = (
        query_X[:, np.newaxis, :]
        - reference_X[np.newaxis, :, :]
    )

    distances = np.sqrt(
        np.sum(differences ** 2, axis=2)
    )

    # Stable ranking by distance, then reference position.
    order = np.argsort(
        distances,
        axis=1,
        kind="stable"
    )

    nearest_indices = order[:, :k]

    neighbour_labels = reference_y[nearest_indices]

    positive_votes = np.sum(
        neighbour_labels == 1,
        axis=1
    )

    # K is odd.
    required_votes = (k + 1) // 2

    return (
        positive_votes >= required_votes
    ).astype(int)


# Fold assignment uses position in the training set.
training_positions = np.arange(train_rows)

fold_ids = training_positions % 5

candidate_ks = list(range(1, 22, 2))

knn_cv_results = []

for k in candidate_ks:

    fold_f1_scores = []

    for fold in range(5):

        validation_mask = fold_ids == fold
        reference_mask = fold_ids != fold

        reference_X = X_train[reference_mask]
        reference_y = y_train[reference_mask]

        validation_X = X_train[validation_mask]
        validation_y = y_train[validation_mask]

        validation_predictions = knn_predict(
            reference_X,
            reference_y,
            validation_X,
            k
        )

        validation_metrics = confusion_metrics(
            validation_y,
            validation_predictions
        )

        fold_f1_scores.append(
            validation_metrics["f1"]
        )

    mean_cv_f1 = float(np.mean(fold_f1_scores))

    knn_cv_results.append((k, mean_cv_f1))


# Write cross-validation results.
with open(
    KNN_CV_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(["K", "mean_cv_f1"])

    for k, score in knn_cv_results:
        writer.writerow([
            k,
            f"{score:.4f}"
        ])


# Ascending candidate order means ties select smaller K.
knn_best_k, best_cv_f1 = max(
    knn_cv_results,
    key=lambda item: item[1]
)

results["knn_best_k"] = knn_best_k

# Refit conceptually using all training rows as reference.
knn_test_predictions = knn_predict(
    X_train,
    y_train,
    X_test,
    knn_best_k
)

knn_test_metrics = confusion_metrics(
    y_test,
    knn_test_predictions
)

for key, value in knn_test_metrics.items():
    results[f"knn_{key}"] = value


# ==================================================
# 2.8 COST-SENSITIVE ALERT THRESHOLD
# ==================================================

FALSE_ALARM_COST = 50
MISSED_POSITIVE_COST = 400


def decision_cost(actual, probabilities, tau):
    predictions = (
        probabilities >= tau
    ).astype(int)

    metrics = confusion_metrics(
        actual,
        predictions
    )

    cost = (
        FALSE_ALARM_COST * metrics["fp"]
        + MISSED_POSITIVE_COST * metrics["fn"]
    )

    return cost, metrics


# Select threshold using training probabilities only.
train_probabilities = sigmoid(
    X_train @ weights + bias
)

best_tau = None
best_train_cost = None

for k in range(1, 20):

    tau = k / 20.0

    cost, _ = decision_cost(
        y_train,
        train_probabilities,
        tau
    )

    # Strict comparison preserves smaller tau on ties.
    if best_train_cost is None or cost < best_train_cost:
        best_train_cost = cost
        best_tau = tau


results["cost_tau"] = best_tau
results["cost_train_at_tau"] = best_train_cost

# Evaluate selected training threshold on test data.
cost_test_at_tau, _ = decision_cost(
    y_test,
    test_probabilities,
    best_tau
)

results["cost_test_at_tau"] = cost_test_at_tau


# Test cost at threshold 0.5.
cost_test_at_05, _ = decision_cost(
    y_test,
    test_probabilities,
    0.5
)

results["cost_test_at_0.5"] = cost_test_at_05


# Cost of flagging nobody.
cost_test_flag_nobody = (
    MISSED_POSITIVE_COST * int(np.sum(y_test))
)

results["cost_test_flag_nobody"] = cost_test_flag_nobody


# ==================================================
# WRITE PART 2 RESULTS
# ==================================================

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")

        else:
            file.write(f"{key} = {value}\n")


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("=" * 65)
print("PHARMEASY HEALTHCARE - PART 2 RESULTS")
print("=" * 65)

for key, value in results.items():

    if isinstance(value, (float, np.floating)):
        print(f"{key:35s} = {value:.4f}")

    else:
        print(f"{key:35s} = {value}")

print("=" * 65)
print("OUTPUT FILES:")
print(RESULTS_FILE)
print(THRESHOLDS_FILE)
print(KNN_CV_FILE)
print("=" * 65)
