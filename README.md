Domain: Healthcare

# PharmEasy Healthcare Decision Intelligence

## How to run

**Requirements:** Python 3.10 or later and NumPy 2.2.6.

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset and reproduce all results:

```bash
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

All generated result files are saved in the `outputs/` directory.

## Part 1 findings

### R1.1 – Estimation and hypothesis testing

The estimated mean care-plan cost is 8.938175 thousand INR, with a 95% confidence interval of [8.302564, 9.573786] thousand INR.

We tested the null hypothesis H0: μ = 8.0 against H1: μ ≠ 8.0 at a 5% significance level.

The calculated z-statistic is 2.8930, which exceeds the two-tailed critical value of 1.96. Therefore, we reject H0.

A Type I error in this context would mean concluding that the true average care-plan cost differs from INR 8,000 when it actually equals INR 8,000.

The difference between the estimated mean and the planning assumption is 0.938175 thousand INR (approximately INR 938.18). This difference is statistically significant and could be practically relevant because it represents an additional estimated cost per care plan.

### R1.2 – Future sample size

The calculated required sample size is 1,293 observations to estimate the average care-plan cost within a margin of error of 0.5 thousand INR (INR 500) at 95% confidence.

The available dataset contains 800 observations, which is 493 fewer than the required sample size. Therefore, the planning team should collect at least 493 additional observations to achieve the specified precision under the same assumptions.

### R1.3 – Gradient descent validation

The absolute difference between the gradient descent slope and the closed-form regression slope is 0.0000 to the reported precision.

Gradient descent reached the closed-form solution because the standardized predictor, full-batch gradient updates and 500 iterations allowed the algorithm to converge to the least-squares minimum.

### R1.4 – Residual funnel and log transformation

The raw regression model has a residual standard deviation ratio of 3.7525 between the upper and lower fitted-value groups.

This indicates a funnel pattern, where the residual spread increases substantially at higher fitted values.

After applying the log-log transformation, the residual standard deviation ratio decreases to 1.0396. This indicates that the log transformation effectively removes the observed funnel pattern and substantially stabilizes residual variability.

### R1.5 – Multiple regression and adjusted R-squared

Using the required mechanical absolute-coefficient threshold of 0.05, the predictors exceeding this threshold are:

1. ln(X4), average bill: coefficient 1.1576.
2. X2, missed visit count: coefficient 0.0940.
3. X3, copay burden ratio: coefficient 0.1067.
4. X6, chronic condition flag: coefficient 0.1518.

The multiple regression model achieves an adjusted R-squared of 0.7403, compared with 0.6561 for the simple log-log regression model.

The improvement of approximately 0.0842 indicates that the additional predictors improve the model's explanatory performance even after adjusting for the number of predictors. Therefore, the additional predictors earn their place in the multiple regression model.

The coefficient threshold is a mechanical selection rule, not a ranking of predictor importance, because the predictors have different measurement scales.

## Part 2 findings

### R2.1 – Learning-rate comparison

After 200 updates, the training losses were:

- Learning rate 1.0: 0.3703
- Learning rate 0.1: 0.3776
- Learning rate 0.01: 0.5091

The learning rate of 0.01 converges relatively slowly, while 1.0 achieves the lowest loss in this experiment; however, excessively large learning rates can cause overshooting or instability.

### R2.2 – Important predictors and odds ratios

The three predictors with the largest absolute logistic regression weights are:

1. Copay burden ratio (X3): 0.9013
2. Months registered (X1): -0.5554
3. Missed visit count (X2): 0.5505

The largest odds ratio is for copay burden ratio (X3), at approximately 2.4629. Holding the other predictors constant, a one-training-standard-deviation increase in copay burden ratio multiplies the odds of a no-show by approximately 2.46.

### R2.3 – Accuracy, precision and recall

The blind model, which predicts zero for every test record, achieves an accuracy of 0.7875 (78.75%). The logistic regression model at a threshold of 0.5 achieves an accuracy of 0.8042 (80.42%).

Accuracy alone is a weak evaluation metric because the dataset is imbalanced, and even a model that predicts no no-shows achieves relatively high accuracy.

When the threshold is lowered from 0.5 to 0.3:

- Recall increases from 0.4706 (47.06%) to 0.7451 (74.51%).
- Precision decreases from 0.5455 (54.55%) to 0.4634 (46.34%).

This demonstrates the trade-off between identifying more potential no-shows and generating additional false alarms.

### R2.4 – Logistic regression versus KNN

The logistic regression model at a threshold of 0.3 achieves an F1-score of 0.5714 and recall of 0.7451.

The KNN model with the selected K of 5 achieves an F1-score of 0.3696 and recall of 0.3333.

Therefore, logistic regression performs better than KNN on both F1-score and recall for this test dataset.

### R2.5 – Cost-sensitive threshold selection and deployment recommendation

The selected logistic regression alert threshold is 0.1500, chosen by minimizing the training-set cost.

The resulting test-set costs are:

- Logistic regression at the selected threshold of 0.15: INR 6,050.
- Logistic regression at threshold 0.5: INR 11,800.
- Flagging nobody: INR 20,400.

I would deploy the logistic regression model with an alert threshold of 0.15, subject to further validation and operational monitoring, because it produces the lowest test-set cost among these evaluated strategies.

The threshold was selected using training data rather than test data to prevent information leakage and preserve the test set as an independent evaluation of the model's performance.

## Part 3 findings

### R3.1 – K-means elbow

The successive WCSS reductions are:

- K=2 to K=3: 659.8282
- K=3 to K=4: 108.0739
- K=4 to K=5: 171.0575
- K=5 to K=6: 45.5466

The elbow rule selects **K=3**, because the reduction from K=3 to K=4 is less than half the preceding reduction. The reductions do not shrink steadily: the pinned starting rows provide a reproducible solution, but not necessarily the global optimum.

### R3.2 – Patient segments

- **A — Established, low-friction users:** 322 records, with an average of 47.5028 months registered, 1.9503 missed visits, and a 0.0186 no-show rate. Action: maintain convenient recurring-care access and use routine reminders rather than intensive follow-up.
- **B — High-bill, moderate-friction users:** 255 records, with an average bill of INR 28.7192 thousand, 6.0706 missed visits, and a 0.2353 no-show rate. Action: offer appointment rescheduling and proactive reminders to reduce missed visits.
- **C — High copay-burden users:** 223 records, with an average copay burden ratio of 0.7118 and a 0.4664 no-show rate. Action: provide clear billing information and convenient appointment confirmation and rescheduling support.

Segment **C** has the highest no-show rate at **0.4664**. These are service-usage profiles, not medical categories.

### R3.3 – Hierarchical clustering

At the largest-gap cut, single linkage produces four clusters: {E1}, {E2, E3, E6}, {E4}, and {E5}. Complete linkage produces two clusters: {E1, E4, E5} and {E2, E3, E6}.

Single linkage merges clusters based on the closest pair of members, so it can connect groups through short links while leaving other members separate. Complete linkage considers the farthest pair, favouring more compact clusters and producing a different grouping.

### R3.4 – PCA

The first principal component is driven positively by X1 (0.5317) and X5 (0.4418), and negatively by X3 (-0.5187), X2 (-0.4100), and X4 (-0.2916). It therefore contrasts longer registration and higher adherence with greater copay burden, missed visits, and average bills.

Three principal components are required to explain at least 80% of the variance; together they explain 0.8616 (86.16%). The mean PC1 scores are A = 1.7773, B = -0.7831, and C = -1.6709. This separates segment A strongly in the positive PC1 direction, while segment C lies furthest in the negative direction.

## Part 4 findings

### R4.1 – Decision-tree split

S4 (X1 ≤ 24 months) is the best candidate because it has the highest information gain (0.1252) and Gini gain (0.0606). A tree grown until every leaf is pure can memorise individual training records and overfit, reducing its ability to generalise. A maximum-depth limit can prevent excessive growth.

### R4.2 – Model comparison

Logistic regression is the preferred structured-data model, with F1 = 0.5714 and recall = 0.7451. The decision stump has the highest recall (0.7647) but a lower F1 (0.5532). KNN performs less effectively, with F1 = 0.3696 and recall = 0.3333.

Combining models did not improve the result: hard voting achieved F1 = 0.5344 and soft voting achieved F1 = 0.5507, both below logistic regression's F1 of 0.5714.

### R4.3 – AdaBoost

After round 1, the two misclassified points (3 and 7) have a combined normalised weight of 0.5000. Round 2 trusts h2 more because its weighted error is lower (0.1667) than h1's error (0.2500), giving h2 a larger alpha of 0.8047 compared with 0.5493.

### R4.4 – Neural-network architecture

**Model A — CNN output shapes and parameters**

| Layer | Output shape | Parameters |
|---|---:|---:|
| Input | 28 × 28 × 1 | 0 |
| CNN Conv1 | 28 × 28 × 6 | 156 |
| MaxPool1 | 14 × 14 × 6 | 0 |
| CNN Conv2 | 10 × 10 × 16 | 2,416 |
| MaxPool2 | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| CNN Dense1 | 32 | 12,832 |
| CNN Dense2 | 10 | 330 |

The complete CNN has 15,734 parameters, compared with 13,002 for the fully connected MLP. The pooling layers reduce the spatial dimensions from 28 × 28 × 6 to 14 × 14 × 6, then to 5 × 5 × 16 before flattening into 400 features.

**Model B — RNN parameter shapes**

- W_xh: 128 × 64
- W_hh: 128 × 128
- W_hy: 16 × 128

The total RNN parameter count is 26,768, including biases. This count does not increase with sequence length because the same recurrent weight matrices are reused at every time step.

For a 10-class image-routing task, I would use ReLU in hidden layers, softmax in the output layer and categorical cross-entropy as the loss function.

In a very deep plain network, gradients can become very small as they propagate backwards, preventing early layers from learning effectively. Skip connections provide a shorter path for information and gradients, helping the network train more effectively.

## Model-choice memo

**To:** Head of Scheduling Operations
**Subject:** Recommendation for the first production no-show prediction model

I recommend deploying the logistic regression model first for structured patient scheduling data, using an alert threshold of 0.15 selected through the training-data cost evaluation.

At the common evaluation threshold of 0.3, logistic regression achieved an F1 score of 0.5714 and recall of 0.7451. The decision stump achieved slightly higher recall of 0.7647 but a lower F1 of 0.5532. KNN achieved an F1 of 0.3696. Combining the models did not improve performance: hard voting achieved an F1 of 0.5344 and soft voting achieved 0.5507.

The Task 2.8 logistic model's test cost at its selected threshold was INR 6,050, compared with INR 11,800 at threshold 0.5 and INR 20,400 when nobody is flagged. These results support using the cost-selected threshold rather than relying on accuracy alone.

One limitation is that the dataset is synthetic and the evaluation uses a single stratified train/test split, so these results do not establish real-world performance.

Model A should be prioritised over Model B for a later development project because image-based scan routing is a defined use case, while the structured scheduling model should first be implemented and evaluated in production.

## Limitations

1. The dataset is synthetic and may not represent the full complexity of real healthcare scheduling behaviour.
2. Model evaluation uses a single stratified train/test split, so performance may vary across different samples.
3. The classification models have not been validated using prospective real-world patient data.
4. The clustering segments describe observed service-use patterns and should not be interpreted as clinical diagnoses.
5. The selected alert threshold depends on the assumed false-alarm and missed-positive costs, which may differ in actual operations.
