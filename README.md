# Python-Machine-Learning
This is Machine Learning Series, we will focus on scikit-learn and tensorflow for supervised and unsupervised machine learning algorithm. 
Main resource comes from book 'Python Machine Learning, 3rd version', with additional real exercises. 
Each portion includes codes, charts and explanations, with notes to help my understanding.
I will add more documentation along the learning.
Happy learning!

<details>
<summary> <b> Week 4 - Model Evaluation and Hyperparameter Tuning (focus on scikit-learn)</b></summary>
[Model Evaluation and Hyperparameter Tuning](<Week 4 - Model Evaluation and Hyperparameter Tuning.ipynb>)
    
#### Why model evaluation
- 'Training error is misleading' - a model that overfits looks perfect on training data and fails on new data
- 'Estimates real-word performance' - evaluate performance on unseen data, check if model is underfit, overfit, or systematically biased on certain segments.
- 'Matches metric to goal' - 'accuracy' is not a good metric for fraud detection since 99% are good transacton only 1% fraud.
#### Why hyperparameter tuning
- 'Defaults are generic, not optimal' - sklearn defaults are picked to be safe across many datasets, not best for yours
- 'Control the bias-variance trade-off' - C, alpha, max_depth, learning rate, dropout all govern over vs. underfitting adn aren't learned from data.
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
    - `roc_curve`: compute the Receiver Operating Characteristic curve by sweeping the decision threshold and returning arrays of false positive rate (FPR) and true positive rate (TPR), and the thresholds themselves. Shows the trade-off between catching positives and raising false alarms across all possible cutoffs. roc_curve(y_true, y_score)
    - `auc`: A general-purpose function that computes the area under curve. acu(fpr, tpr).
    - `roc_auc_score`: A general-purpose function that computes the area under curve. acu(y_true, y_score).
    - `precision_recall_curve`: Computes precision and recall at every possible decision threshold, returning three arrays: precision, recall, and thresholds. especially valuable when classes are imbalanced, since it doesn't inflated by the large true-negative count. precision_recall_curve(y_true, y_score)
- ``Class imbalance``
    - class_weight='balanced': automatically reweights training samples to compensate for calss imbalances - without resampling the data. Instead of treating every training example equally in the loss function, it sccales each class's contribution inversely proportional to its frequency. Misclassifying a minority-class example becomes "more expensive" than misclassifying a majority-class one, so the model stops ignoring the rare class.
#### How to process hyperparameter tuning
- ``Hyperparameter tuning:``
    - `GridSearchSV`: search best hyperparameters by all combinations
    - `RandomizedSearchCV` - better when search space is large

</details>
