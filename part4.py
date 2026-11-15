
import csv
import math
import os
import numpy as np

INPUT_FILE = "data/records.csv"
OUTPUT_DIR = "outputs"


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_data():
    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 800:
        raise ValueError(f"Expected 800 records, found {len(rows)}")

    columns = [
        "months_registered",
        "missed_visit_count_12m",
        "copay_burden_ratio",
        "avg_bill_inr_thousands",
        "adherence_score",
        "chronic_condition_flag",
        "insured_flag",
    ]

    X = np.column_stack([
        np.array([float(row[col]) for row in rows])
        for col in columns
    ])

    y = np.array([float(row["no_show"]) for row in rows])
    return X, y


# =========================================================
# 2. SAME STRATIFIED SPLIT AS PART 2
# =========================================================

def make_split(y):
    negative = np.where(y == 0)[0]
    positive = np.where(y == 1)[0]

    n_negative = (7 * len(negative)) // 10
    n_positive = (7 * len(positive)) // 10

    train_idx = np.sort(np.concatenate([
        negative[:n_negative],
        positive[:n_positive]
    ]))

    test_idx = np.sort(np.concatenate([
        negative[n_negative:],
        positive[n_positive:]
    ]))

    return train_idx, test_idx


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X, y, learning_rate=0.1, updates=1000):
    n, p = X.shape
    weights = np.zeros(p)
    bias = 0.0

    for _ in range(updates):
        probabilities = sigmoid(X @ weights + bias)
        error = probabilities - y

        dw = (X.T @ error) / n
        db = np.sum(error) / n

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias


def confusion_metrics(actual, predicted):
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    def divide(a, b):
        return a / b if b else 0.0

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    f1 = divide(2 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def knn_positive_share(reference_X, reference_y, query_X, k):
    differences = (
        query_X[:, np.newaxis, :]
        - reference_X[np.newaxis, :, :]
    )

    distances = np.sqrt(np.sum(differences ** 2, axis=2))

    order = np.argsort(
        distances,
        axis=1,
        kind="stable"
    )

    nearest = order[:, :k]
    labels = reference_y[nearest]

    return np.sum(labels == 1, axis=1) / k


def entropy(labels):
    if len(labels) == 0:
        return 0.0

    p1 = np.mean(labels == 1)
    p0 = 1.0 - p1

    result = 0.0

    for p in (p0, p1):
        if p > 0:
            result -= p * math.log2(p)

    return result


def gini(labels):
    if len(labels) == 0:
        return 0.0

    p1 = np.mean(labels == 1)
    p0 = 1.0 - p1

    return 1.0 - p0 ** 2 - p1 ** 2


def weighted_impurity(parent, yes, no, function):
    n = len(parent)

    return (
        len(yes) / n * function(yes)
        + len(no) / n * function(no)
    )


def write_results(results):
    path = os.path.join(OUTPUT_DIR, "part4_results.txt")

    with open(path, "w", encoding="utf-8") as f:
        for key, value in results.items():
            if isinstance(value, (float, np.floating)):
                f.write(f"{key} = {value:.4f}\n")
            else:
                f.write(f"{key} = {value}\n")


# =========================================================
# 4. MAIN EXECUTION
# =========================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_raw, y = load_data()

    train_idx, test_idx = make_split(y)

    X_train_raw = X_raw[train_idx]
    X_test_raw = X_raw[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    if (
        len(y_train),
        int(np.sum(y_train)),
        len(y_test),
        int(np.sum(y_test))
    ) != (560, 119, 240, 51):
        raise ValueError("Part 2 split checkpoint mismatch.")

    # Standardise using training data only.
    train_mean = np.mean(X_train_raw, axis=0)
    train_sd = np.std(X_train_raw, axis=0, ddof=1)

    if np.any(train_sd == 0):
        raise ValueError("Zero standard deviation found.")

    X_train = (X_train_raw - train_mean) / train_sd
    X_test = (X_test_raw - train_mean) / train_sd

    results = {}

    # =====================================================
    # 4.1 ENTROPY AND GINI SPLITS
    # =====================================================

    conditions = {
        "S1": X_train_raw[:, 5] == 1,
        "S2": X_train_raw[:, 6] == 1,
        "S3": X_train_raw[:, 2] > 0.5,
        "S4": X_train_raw[:, 0] <= 24,
    }

    root_entropy = entropy(y_train)
    root_gini = gini(y_train)

    results["root_entropy"] = root_entropy
    results["root_gini"] = root_gini

    info_gains = {}
    gini_gains = {}
    split_rows = []

    for name, mask in conditions.items():

        yes = y_train[mask]
        no = y_train[~mask]

        info_gain = (
            root_entropy
            - weighted_impurity(y_train, yes, no, entropy)
        )

        gini_gain = (
            root_gini
            - weighted_impurity(y_train, yes, no, gini)
        )

        info_gains[name] = info_gain
        gini_gains[name] = gini_gain

        split_rows.append([
            name,
            len(yes),
            int(np.sum(yes)),
            len(no),
            int(np.sum(no)),
            f"{info_gain:.4f}",
            f"{gini_gain:.4f}",
        ])

    best_info = max(info_gains, key=info_gains.get)
    best_gini = max(gini_gains, key=gini_gains.get)

    results["best_split_by_info_gain"] = best_info
    results["best_split_by_gini_gain"] = best_gini

    with open(
        os.path.join(OUTPUT_DIR, "part4_splits.csv"),
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "split",
            "n_yes",
            "positives_yes",
            "n_no",
            "positives_no",
            "info_gain",
            "gini_gain"
        ])

        writer.writerows(split_rows)

    # =====================================================
    # 4.2 LOGISTIC REGRESSION
    # =====================================================

    weights, bias = fit_logistic(
        X_train,
        y_train,
        learning_rate=0.1,
        updates=1000
    )

    logistic_test_prob = sigmoid(
        X_test @ weights + bias
    )

    # =====================================================
    # 4.2 KNN CROSS VALIDATION
    # =====================================================

    fold_ids = np.arange(len(y_train)) % 5

    cv_results = []

    for k in range(1, 22, 2):

        fold_scores = []

        for fold in range(5):

            validation_mask = fold_ids == fold
            reference_mask = ~validation_mask

            probabilities = knn_positive_share(
                X_train[reference_mask],
                y_train[reference_mask],
                X_train[validation_mask],
                k
            )

            predictions = (
                probabilities >= (k + 1) / (2 * k)
            ).astype(int)

            metrics = confusion_metrics(
                y_train[validation_mask],
                predictions
            )

            fold_scores.append(metrics["f1"])

        cv_results.append((k, np.mean(fold_scores)))

    # Select smaller K in the event of a tie.
    best_k, best_cv_f1 = max(
        cv_results,
        key=lambda item: item[1]
    )

    knn_test_prob = knn_positive_share(
        X_train,
        y_train,
        X_test,
        best_k
    )

    knn_test_pred = (
        knn_test_prob >= (best_k + 1) / (2 * best_k)
    ).astype(int)

    results["knn_best_k"] = int(best_k)

    # =====================================================
    # 4.2 DECISION STUMP
    # =====================================================

    best_mask = conditions[best_info]

    stump_rate_yes = float(np.mean(y_train[best_mask]))
    stump_rate_no = float(np.mean(y_train[~best_mask]))

    test_conditions = {
        "S1": X_test_raw[:, 5] == 1,
        "S2": X_test_raw[:, 6] == 1,
        "S3": X_test_raw[:, 2] > 0.5,
        "S4": X_test_raw[:, 0] <= 24,
    }

    stump_test_prob = np.where(
        test_conditions[best_info],
        stump_rate_yes,
        stump_rate_no
    )

    results["stump_rate_yes"] = stump_rate_yes
    results["stump_rate_no"] = stump_rate_no

    # All three base models use threshold 0.3.
    logistic_pred = (
        logistic_test_prob >= 0.3
    ).astype(int)

    stump_pred = (
        stump_test_prob >= 0.3
    ).astype(int)

    hard_vote_pred = (
        logistic_pred + knn_test_pred + stump_pred >= 2
    ).astype(int)

    soft_vote_prob = (
        logistic_test_prob
        + knn_test_prob
        + stump_test_prob
    ) / 3.0

    soft_vote_pred = (
        soft_vote_prob >= 0.3
    ).astype(int)

    models = [
        ("logistic", logistic_pred),
        ("knn", knn_test_pred),
        ("stump", stump_pred),
        ("hard_vote", hard_vote_pred),
        ("soft_vote", soft_vote_pred),
    ]

    voting_rows = []

    for name, prediction in models:

        metrics = confusion_metrics(y_test, prediction)

        voting_rows.append([
            name,
            metrics["tp"],
            metrics["fp"],
            metrics["tn"],
            metrics["fn"],
            f'{metrics["precision"]:.4f}',
            f'{metrics["recall"]:.4f}',
            f'{metrics["f1"]:.4f}',
        ])

    with open(
        os.path.join(OUTPUT_DIR, "part4_voting.csv"),
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "model",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "f1"
        ])

        writer.writerows(voting_rows)

    # =====================================================
    # 4.3 ADABOOST - TWO ROUNDS
    # =====================================================

    true_y = np.array([
        1, 1, 1, -1, -1, -1, 1, -1
    ], dtype=float)

    h1 = np.array([
        1, 1, -1, -1, -1, -1, -1, -1
    ], dtype=float)

    h2 = np.array([
        -1, 1, 1, -1, 1, -1, 1, -1
    ], dtype=float)

    sample_weights = np.full(8, 1.0 / 8.0)

    hypotheses = [h1, h2]
    alphas = []
    adaboost_rows = []

    for round_no, hypothesis in enumerate(hypotheses, start=1):

        error = float(
            np.sum(sample_weights[hypothesis != true_y])
        )

        if not 0 < error < 1:
            raise ValueError("Invalid AdaBoost error.")

        alpha = 0.5 * math.log((1 - error) / error)

        unnormalised = (
            sample_weights
            * np.exp(-alpha * true_y * hypothesis)
        )

        updated = unnormalised / np.sum(unnormalised)

        results[f"round{round_no}_error"] = error
        results[f"round{round_no}_alpha"] = alpha

        for i in range(8):
            adaboost_rows.append([
                round_no,
                i + 1,
                f"{sample_weights[i]:.4f}",
                f"{unnormalised[i]:.4f}",
                f"{updated[i]:.4f}",
            ])

        sample_weights = updated
        alphas.append(alpha)

    ensemble_score = (
        alphas[0] * h1 + alphas[1] * h2
    )

    ensemble_prediction = np.where(
        ensemble_score >= 0,
        1,
        -1
    )

    results["ensemble_accuracy"] = float(
        np.mean(ensemble_prediction == true_y)
    )

    with open(
        os.path.join(OUTPUT_DIR, "part4_adaboost.csv"),
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "round",
            "point",
            "weight_before",
            "weight_unnormalised",
            "weight_after"
        ])

        writer.writerows(adaboost_rows)

    # =====================================================
    # 4.4 OUT-OF-BAG FRACTION
    # =====================================================

    results["oob_fraction"] = (
        1.0 - 1.0 / 560.0
    ) ** 560

    # =====================================================
    # 4.5 ARCHITECTURE PARAMETER COUNTS
    # =====================================================

    architecture = [
        ("mlp_hidden1", 784 * 16 + 16),
        ("mlp_hidden2", 16 * 16 + 16),
        ("mlp_output", 16 * 10 + 10),

        ("cnn_conv1", 5 * 5 * 1 * 6 + 6),
        ("cnn_conv2", 5 * 5 * 6 * 16 + 16),
        ("cnn_dense1", 5 * 5 * 16 * 32 + 32),
        ("cnn_dense2", 32 * 10 + 10),
    ]

    results["mlp_total"] = sum(
        value for _, value in architecture[:3]
    )

    results["cnn_total"] = sum(
        value for _, value in architecture[3:]
    )

    results["fc_layer_params"] = (
        128 * 128 * 3 * 1024 + 1024
    )

    results["conv_layer_params"] = (
        3 * 3 * 3 * 64 + 64
    )

    results["rnn_total"] = (
        64 * 128
        + 128 * 128
        + 128 * 16
        + 128
        + 16
    )

    with open(
        os.path.join(OUTPUT_DIR, "part4_architecture.csv"),
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["layer", "parameters"])
        writer.writerows(architecture)

    # =====================================================
    # 5. WRITE RESULTS
    # =====================================================

    write_results(results)

    print("=" * 60)
    print("PHARMEASY HEALTHCARE - PART 4")
    print("=" * 60)

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            print(f"{key:35s} = {value:.4f}")
        else:
            print(f"{key:35s} = {value}")

    print("=" * 60)
    print("Generated output files:")

    for filename in [
        "part4_results.txt",
        "part4_splits.csv",
        "part4_voting.csv",
        "part4_adaboost.csv",
        "part4_architecture.csv"
    ]:
        print(os.path.join(OUTPUT_DIR, filename))

    print("=" * 60)


if __name__ == "__main__":
    main()
