## Logistic Regression and Model Evaluation

<details>
    
<summary><b>Week 4.1 - Model Evaluation and Hyperparameter Tuning (scikit-learn)</b></summary>

[Model Evaluation and Hyperparameter Tuning: scikit-learn](Week4-Model_Evaluation_and_Hyperparameter_Tuning.ipynb)

#### Why model evaluation
- `Training error is misleading` - a model that overfits looks perfect on training data and fails on new data
- `Estimates real-word performance` - evaluate performance on unseen data, check if model is underfit, overfit, or systematically biased on certain segments.
- `Matches metric to goal` - 'accuracy' is not a good metric for fraud detection since 99% are good transacton only 1% fraud.

#### Why hyperparameter tuning
- `Defaults are generic, not optimal` - sklearn defaults are picked to be safe across many datasets, not best for yours
- `Control the bias-variance trade-off` - C, alpha, max_depth, learning rate, dropout all govern over vs. underfitting adn aren't learned from data.

#### How to evaluate model
- ``pipeline, make_pipeline:``
    - Chaining preprocessing + estimator into one object. Critical for avoiding data leakage during cross-validation.
- ``cross_validation:``
    - `StratifiedKFold`: split data by folds, then train and evaluate each of them
    - `cross_val_score`: split data by folds automatically, then return the score (R^2 for regression, accuracy for classification)
- ``Learning curves & validation curves``
    - `learing_curve`: performance vs. training set size -> diagnose bias vs. variance
    - `LearningCurveDisplay`: display the learing curve
    - `validation_curve`: performance vs. a single hyperparameter -> diagnose under/overfitting for that param
- ``Evaluation metrics``
    - `confusion_matrix`: A table of counts comparing predicted vs. actual labels. [[TP, FN],[FP, TN]]
    - `ConfusionMatrixDisplay`: A visualization wrapper that renders a confusion matrix as a heatmap with annotated cell counts.
    - `precision_score`: the fraction of predicted postives that are actually positive. TP / (TP / FP)
    - `recall_score`: the fraction of actual positives that the model correctly identified. TP / (TP + FN)
    - `f1_score`: the harmonic mean of precision and recall 2 * (Precision * Recall) / (Precision + Recall)
    - `roc_curve`: compute the Receiver Operating Characteristic curve by sweeping the decision threshold and returning arrays of false positive rate (FPR) and true positive rate (TPR), and the thresh[...]
    - `auc`: A general-purpose function that computes the area under curve. acu(fpr, tpr).
    - `roc_auc_score`: A general-purpose function that computes the area under curve. acu(y_true, y_score).
    - `precision_recall_curve`: Computes precision and recall at every possible decision threshold, returning three arrays: precision, recall, and thresholds. especially valuable when classes are imba[...]
- ``Class imbalance``
    - class_weight='balanced': automatically reweights training samples to compensate for calss imbalances - without resampling the data. Instead of treating every training example equally in the loss[...]

#### How to process hyperparameter tuning
- ``Hyperparameter tuning:``
    - `GridSearchSV`: search best hyperparameters by all combinations
    - `RandomizedSearchCV` - better when search space is large

</details>

<details>
    
<summary><b>Week 4.2 - Hyperparameter Tuning (TensorFlow)</b></summary>

[Hyperparameter Tuning: TensorFlow](Week4-TensorFlow_Tuning.ipynb)

#### What is keras_tuner
- it is a hyperparameter optimization library built specifically for Keras/TensorFlow models. It automates the process of searching for the best hyperparameter values rather than you manually tring combination.

#### Why use keras_tuner
- `Systematic search` over a defined space, with reproducibility via seeds
- `Early stopping` of bad trials, so compute goes to promisign configs instead of being wasted finishing a clearly-bad run
- `Result tracking`, every trial's hyperparameters, metrics, and checkpoints are saved to disk, so you can resume, inspect, and pull tuner.get_best_hyperparameters() afterward.
- `Integrationg with Keras callbacks, EarlyStopping, TensorBoard` work normally inside trials.

#### When use it
- For a tuned MLP, CNN, or anything with depth/width/learning-rate/regularization choices interacting, the tuner usually finds something better than manually pick.

#### What search algorithms
- `RandomSearch:` samples configs randomly
- `Hyperbank:` runs many configs for a few epchs, kills the bad ones early, gives more budget to survivors. Very efficient when training is expensive.
- `BayesianOptimization:` builds a probabilistic model of which hyperparameters tend to work and samples promising regions.
- `GridSearch:` run ever number, usually only practical for tiny search spaces.

</details>

<details>
    
<summary><b>Week 4.3 - roc_curve and pr_curve </b></summary>

##### What is `roc_curve`
- probability the model ranks a random positive above a random negative, range from 0 to 1, 0.5 is random.
- at each cutoff, how many positives have I caught vs. how many negatives have I falsely caught.
  
##### What is `pr_curve`
- pr_curve (precision-recall curve) plots precision (y-axis) against recall (x-axis) across all classification thresholds (from 1 to 0) for a binary classifier.
    - precision = TP / (TP + FP), of predicted positives, how many are correct
    - recall = TP / (TP + FN), of actual postiives, how many you caught
- As we sweep the decision threshold from high to low we trade precision fro recall. The summary metrics is average precision (AP) or area under the pr_curve.
- `High threshold` -> few predicted positive -> high precision (capture correctly with strict rules), low recall. Far left (recall around 0), we set a very high threshold like 0.99, we only flag examles we are extremely sure about, we catch very few positives but almost everything we flag is correct (high precision).
- `low threshold` -> many predicted positive -> low precision (predict wrongly with less strict rules), high recall. Far right (recall=1.0), threshold near 0. we flag almost everything as positive. we catch all the positive but our precision collapses to the base rate because we are flagging a ton of negatives too.
- pr_curve is prefered over ROC curves on imbalanced datasets (e.g. fraud detection), because ROC's false positve rate uses the large true-negative count in its denominator and stays optimistic when positives are rare, pr_curve only involve the positive class.
- base rate = Positive / (Positive + Negative)
- pr_curve for random is a horizontal line at y = base rate.
    - At anty threshold, the set of instances it flags are positive is just a random sample of all instances. In any random sample, the fraction that are actually positive equals the overall positive rate in the data - the base rate. So precision = base rate no matter where you set the threshold.
    - Chaning the threshold changes how many you flag (and thus recall sweeps from 0 to 1), but it doens't change the purity of what you flag, since you're always grabbing a representative random slice.

##### How to use `pr_curve`
- Pick the operating region that matches the cost function. If false positives are expensive (blocking legit Strip transactions annoys merchants), operate top-left: high precision, lower recall. If false negatives are catastrophic (missing fraud = chargeback losses), operate bottom-right: high recall, accept more FPs.
- The AP number is the area under the whole curve - a single summary across all thresholds. Userful for comparing models, less useful for picking an operating point.

##### `pr_curve` - Why precision rate >= equal base rate (because the TN)
- Precision at recall=1.0 = base rate only when the worst positive ranks dead last (model has zero ability to separate that hardest positive from negatives).
- Precision at recall=1.0 > base rate whenever at least one negative ranks below the weakest positive — i.e., the model has some discriminative power even at the bottom.

#### What are the difference between roc_curve and pr_curve
`Point 1:`
- roc_curve's random base line is always 0.5, regardless of class balance.
    - A random classifier assigns scores with no relationship to the true labels. At any threshold, it flags a given fraction of all examples regardless of class — so it captures positives and negatives at the same rate. That means True Positive Rate equals False Positive Rate at every threshold, which traces the diagonal line from (0,0) to (1,1). The area under that diagonal is exactly 0.5.
    - Intuitively: AUC equals the probability that the model ranks a random positive above a random negative. If scores are random, that's a coin flip — 0.5.
    - Two caveats worth knowing:
        - The 0.5 baseline is independent of class balance. This is a key advantage of ROC over precision-recall, where the baseline shifts with the positive rate.
        - Below 0.5 isn't "worse than random" in a useful sense — it means the classifier is anti-correlated, so inverting its predictions gives you AUC > 0.5.
- pr_curve:
    - AP's random baseline equals the positive class prevalence. Precision's denominator is "things you predicted positive," which includes false positives drawn from the negative class. A random classifier flagging things at rate r will have precision ≈ base rate at every recall level (because among randomly flagged items, the fraction that are truly positive equals the overall positive rate). So the PR curve for random is a horizontal line at y = base rate, and AP ≈ base rate.
    - Why this matters in practice: if someone tells you "my model has AUC-ROC = 0.85" you immediately know it's well above the 0.5 random floor. If someone tells you "my model has AP = 0.30" you have no idea if it's good until you ask "what's the positive rate?" AP = 0.30 with 1% positives is brilliant. AP = 0.30 with 40% positives is barely better than random.
      
`Point 2:`
- AUC-ROC is comparable across datasets because the baseline (0.5) and ceiling (1.0) are fixed. If model A gets AUC = 0.82 on dataset X and model B gets AUC = 0.79 on dataset Y, you can meaningfully say A scored higher — both are measured against the same scale.
- AP is not comparable across datasets because the floor moves with the data. Model A getting AP = 0.60 on a 30% positive dataset and Model B getting AP = 0.40 on a 5% positive dataset — which is better? Naively A looks better, but:
    - A's lift over random: 0.60 / 0.30 = 2.0×
    - B's lift over random: 0.40 / 0.05 = 8.0×
    - B is actually the much stronger model relative to its baseline. The raw AP numbers are misleading.
- Workarounds for AP comparison:
    - Report lift (AP / base rate). Normalizes away the prevalence.
    - Report normalized AP: (AP − base rate) / (1 − base rate). Scales the metric to [0, 1] where 0 = random and 1 = perfect, regardless of base rate.
    - Just always report base rate alongside AP. Cheap and honest — readers can do the math themselves.

#### How to calculate AP (Average Precision) - Weighted average precesion
Here's a worked example with 5 predictions ranked by confidence. Say there are **3 actual positives** total.
note: score is the model's confidence output fo a single prediciton, every prediction has its own score, when threshold = 0.95 -> only rank 1 is called positive, when threshold = 0.88 -> ranks 1-2 are psotive.
| Rank | Score | True label | TP count | FP count | Precision | Recall |
|------|-------|-----------|----------|----------|-----------|--------|
| 1 | 0.95 | Positive ✓ | 1 | 0 | 1/1 = 1.00 | 1/3 = 0.33 |
| 2 | 0.88 | Positive ✓ | 2 | 0 | 2/2 = 1.00 | 2/3 = 0.67 |
| 3 | 0.71 | Negative ✗ | 2 | 1 | 2/3 = 0.67 | 2/3 = 0.67 |
| 4 | 0.63 | Positive ✓ | 3 | 1 | 3/4 = 0.75 | 3/3 = 1.00 |
| 5 | 0.52 | Negative ✗ | 3 | 2 | 3/5 = 0.60 | 3/3 = 1.00 |

**Apply the formula** AP = Σ (Rₙ − Rₙ₋₁) · Pₙ, starting with R₀ = 0:

| Rank | Rₙ − Rₙ₋₁ | Pₙ | Contribution |
|------|-----------|-----|-------------|
| 1 | 0.33 − 0 = 0.33 | 1.00 | 0.33 |
| 2 | 0.67 − 0.33 = 0.33 | 1.00 | 0.33 |
| 3 | 0.67 − 0.67 = 0 | 0.67 | 0 |
| 4 | 1.00 − 0.67 = 0.33 | 0.75 | 0.25 |
| 5 | 1.00 − 1.00 = 0 | 0.60 | 0 |

**AP = 0.33 + 0.33 + 0 + 0.25 + 0 = 0.91**

Notice two things:

- Recall only increases when you hit a **true positive**, so the negatives at ranks 3 and 5 contribute zero (their recall step is 0). This is why AP is often described as "average the precision each time you hit a true positive": 1.00, 1.00, and 0.75 → mean = 0.917, which matches (small rounding aside).
- The model is penalized for that negative at rank 3 — it dropped precision to 0.75 by the time the third true positive appeared at rank 4. A perfect ranking (3 positives at the top) would give precisions 1.00, 1.00, 1.00 → AP = 1.00.
        
</details>

