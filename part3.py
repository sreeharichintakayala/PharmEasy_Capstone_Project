
"""
PharmEasy Healthcare Decision Intelligence
Part 3: Segmentation and Dimensionality Reduction

Run:
    python part3.py

Input:
    data/records.csv

Outputs:
    outputs/part3_results.txt
    outputs/part3_segments.csv
    outputs/part3_hierarchical.csv
"""

import csv
import math
import os
import numpy as np


# ==================================================
# 1. LOAD DATA
# ==================================================

INPUT_FILE = "data/records.csv"
OUTPUT_DIR = "outputs"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR, "part3_results.txt"
)
SEGMENTS_FILE = os.path.join(
    OUTPUT_DIR, "part3_segments.csv"
)
HIERARCHICAL_FILE = os.path.join(
    OUTPUT_DIR, "part3_hierarchical.csv"
)

with open(INPUT_FILE, "r", newline="") as file:
    rows = list(csv.DictReader(file))

if len(rows) != 800:
    raise ValueError(f"Expected 800 rows, found {len(rows)}")


def column(name):
    return np.array([float(r[name]) for r in rows])


# X1 to X5 only: five continuous predictors
X_original = np.column_stack([
    column("months_registered"),
    column("missed_visit_count_12m"),
    column("copay_burden_ratio"),
    column("avg_bill_inr_thousands"),
    column("adherence_score")
])

outcome = column("no_show")
value = column("care_plan_cost_inr_thousands")

n = len(X_original)
results = {}

# ==================================================
# 3.1 STANDARDISE X1 TO X5
# ==================================================

feature_mean = np.mean(X_original, axis=0)
feature_sd = np.std(X_original, axis=0, ddof=1)

if np.any(feature_sd == 0):
    raise ValueError("A feature has zero standard deviation.")

Z = (X_original - feature_mean) / feature_sd

# ==================================================
# 3.1 K-MEANS FROM SCRATCH
# ==================================================

def kmeans(X, k):
    """
    K-means with pinned initial rows.

    Initial indices:
    0, floor(800/K), 2*floor(800/K), ...

    Assignment ties go to the lower-numbered centroid.
    Empty centroids remain unchanged.
    Stop at first unchanged assignment pass.
    """

    count = len(X)
    step = count // k

    initial_indices = [
        j * step for j in range(k)
    ]

    centroids = X[initial_indices].copy()

    # Initial labels deliberately differ from all assignments,
    # ensuring that the first assignment pass is performed.
    labels = np.full(count, -1, dtype=int)

    while True:

        previous_labels = labels.copy()

        # Squared Euclidean distances.
        differences = (
            X[:, np.newaxis, :]
            - centroids[np.newaxis, :, :]
        )

        squared_distances = np.sum(
            differences ** 2, axis=2
        )

        # np.argmin selects the first centroid on ties.
        labels = np.argmin(
            squared_distances, axis=1
        )

        # Stop on the first pass with no changed assignments.
        if np.array_equal(labels, previous_labels):
            break

        # Update centroids.
        for cluster in range(k):
            members = X[labels == cluster]

            if len(members) > 0:
                centroids[cluster] = np.mean(
                    members, axis=0
                )

    # WCSS in standardised feature space.
    final_differences = X - centroids[labels]

    wcss = float(np.sum(
        final_differences ** 2
    ))

    return labels, centroids, wcss


kmeans_results = {}

for k in range(2, 7):

    labels, centroids, wcss = kmeans(Z, k)

    kmeans_results[k] = {
        "labels": labels,
        "centroids": centroids,
        "wcss": wcss
    }

    results[f"wcss_k{k}"] = wcss


# ==================================================
# 3.2 ELBOW METHOD
# ==================================================

elbow_k = None

for k in (3, 4, 5):

    current_drop = (
        kmeans_results[k]["wcss"]
        - kmeans_results[k + 1]["wcss"]
    )

    previous_drop = (
        kmeans_results[k - 1]["wcss"]
        - kmeans_results[k]["wcss"]
    )

    if current_drop < 0.5 * previous_drop:
        elbow_k = k
        break

if elbow_k is None:
    raise RuntimeError("No elbow K found in K=3,4,5.")

results["elbow_k"] = elbow_k


# ==================================================
# 3.3 PROFILE K=3 SEGMENTS
# ==================================================

labels3 = kmeans_results[3]["labels"]

cluster_ids = list(range(3))

# Order clusters by mean original X3 (copay burden ratio).
cluster_ids.sort(
    key=lambda cluster: np.mean(
        X_original[labels3 == cluster, 2]
    )
)

segment_names = ["A", "B", "C"]

segment_indices = {}

for name, cluster in zip(segment_names, cluster_ids):

    indices = np.where(labels3 == cluster)[0]

    segment_indices[name] = indices

    results[f"size_segment_{name}"] = len(indices)


with open(
    SEGMENTS_FILE, "w", newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "segment",
        "size",
        "mean_X1",
        "mean_X2",
        "mean_X3",
        "mean_X4",
        "mean_X5",
        "outcome_rate",
        "mean_value"
    ])

    for name in segment_names:

        indices = segment_indices[name]

        feature_means = np.mean(
            X_original[indices], axis=0
        )

        outcome_rate = np.mean(
            outcome[indices]
        )

        mean_value = np.mean(
            value[indices]
        )

        writer.writerow([
            name,
            len(indices),
            *(f"{v:.4f}" for v in feature_means),
            f"{outcome_rate:.4f}",
            f"{mean_value:.4f}"
        ])


# ==================================================
# 3.4 HIERARCHICAL CLUSTERING
# ==================================================

cohort_names = ["E1", "E2", "E3", "E4", "E5", "E6"]

distance_matrix = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
], dtype=float)


def cluster_distance(cluster_a, cluster_b, linkage):
    """Calculate linkage distance between two clusters."""

    pair_distances = [
        distance_matrix[i, j]
        for i in cluster_a
        for j in cluster_b
    ]

    if linkage == "single":
        return min(pair_distances)

    if linkage == "complete":
        return max(pair_distances)

    if linkage == "average":
        return sum(pair_distances) / len(pair_distances)

    raise ValueError("Unknown linkage")


def cluster_text(cluster):
    """Write cluster members in ascending cohort order."""

    return "+".join(
        cohort_names[i]
        for i in sorted(cluster)
    )


hierarchical_rows = []
cut_counts = {}

for linkage in ("single", "complete", "average"):

    clusters = [
        [i] for i in range(6)
    ]

    heights = []

    for merge_no in range(1, 6):

        best_distance = float("inf")
        best_pair = None

        # Search every pair of current clusters.
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):

                d = cluster_distance(
                    clusters[i],
                    clusters[j],
                    linkage
                )

                if d < best_distance:
                    best_distance = d
                    best_pair = (i, j)

        i, j = best_pair

        cluster_a = clusters[i]
        cluster_b = clusters[j]

        # cluster_a must contain the smaller minimum member.
        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a

        heights.append(best_distance)

        hierarchical_rows.append([
            linkage,
            merge_no,
            cluster_text(cluster_a),
            cluster_text(cluster_b),
            f"{best_distance:.4f}"
        ])

        merged = sorted(cluster_a + cluster_b)

        # Remove the two original clusters.
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (i, j)
        ]

        clusters.append(merged)

        # Keep cluster order deterministic.
        clusters.sort(key=lambda cluster: min(cluster))

    # Largest gap between consecutive merge heights.
    gaps = [
        heights[i + 1] - heights[i]
        for i in range(4)
    ]

    largest_gap_index = int(np.argmax(gaps))

    merges_below_cut = largest_gap_index + 1

    clusters_at_cut = 6 - merges_below_cut

    cut_counts[linkage] = clusters_at_cut


with open(
    HIERARCHICAL_FILE, "w", newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "linkage",
        "merge_no",
        "cluster_a",
        "cluster_b",
        "height"
    ])

    writer.writerows(hierarchical_rows)


results["single_clusters_at_cut"] = cut_counts["single"]
results["complete_clusters_at_cut"] = cut_counts["complete"]
results["average_clusters_at_cut"] = cut_counts["average"]


# ==================================================
# 3.5 PCA: SEVEN STEPS
# ==================================================

# Step 1: Mean vector of standardised data.
pca_mean = np.mean(Z, axis=0)

# Step 2: Centre the data.
X_centered = Z - pca_mean

# Step 3: Covariance matrix.
covariance_matrix = (
    X_centered.T @ X_centered
) / (n - 1)

# Step 4: Eigen-decomposition.
eigenvalues, eigenvectors = np.linalg.eigh(
    covariance_matrix
)

# Step 5: Sort by descending eigenvalue.
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Apply sign convention:
# Largest absolute loading in each eigenvector must be positive.
for j in range(eigenvectors.shape[1]):

    largest_index = np.argmax(
        np.abs(eigenvectors[:, j])
    )

    if eigenvectors[largest_index, j] < 0:
        eigenvectors[:, j] *= -1


# Step 6: Keep the top two eigenvectors.
W = eigenvectors[:, :2]

# Step 7: Project data.
projections = X_centered @ W

eigenvalue_sum = float(np.sum(eigenvalues))

explained_ratios = eigenvalues / eigenvalue_sum

cumulative_ratios = np.cumsum(
    explained_ratios
)

components_for_80pct = int(
    np.argmax(cumulative_ratios >= 0.80) + 1
)

results["eigenvalue_sum"] = eigenvalue_sum

for i in range(5):

    results[f"eigenvalue_{i + 1}"] = eigenvalues[i]

    results[f"explained_ratio_{i + 1}"] = (
        explained_ratios[i]
    )

    results[f"cumulative_ratio_{i + 1}"] = (
        cumulative_ratios[i]
    )

results["components_for_80pct"] = components_for_80pct


# PC1 and PC2 loadings.
for i in range(5):

    results[f"pc1_loading_X{i + 1}"] = W[i, 0]

    results[f"pc2_loading_X{i + 1}"] = W[i, 1]


# Projections of first three rows.
for row_index in range(3):

    results[f"row{row_index + 1}_pc1"] = (
        projections[row_index, 0]
    )

    results[f"row{row_index + 1}_pc2"] = (
        projections[row_index, 1]
    )


# Mean PC1 score for each segment.
for name in segment_names:

    indices = segment_indices[name]

    results[f"mean_pc1_segment_{name}"] = (
        np.mean(projections[indices, 0])
    )


# ==================================================
# WRITE RESULTS
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(
    RESULTS_FILE, "w", encoding="utf-8"
) as file:

    for key, val in results.items():

        if isinstance(val, (float, np.floating)):
            file.write(f"{key} = {val:.4f}\n")

        else:
            file.write(f"{key} = {val}\n")


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("=" * 65)
print("PHARMEASY HEALTHCARE - PART 3 RESULTS")
print("=" * 65)

for key, val in results.items():

    if isinstance(val, (float, np.floating)):
        print(f"{key:35s} = {val:.4f}")

    else:
        print(f"{key:35s} = {val}")

print("=" * 65)
print("OUTPUT FILES:")
print(RESULTS_FILE)
print(SEGMENTS_FILE)
print(HIERARCHICAL_FILE)
print("=" * 65)
