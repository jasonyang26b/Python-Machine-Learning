# Python-Machine-Learning
This is Machine Learning Series, we will focus on sklearn and tensorflow for supervised and unsupervised machine learning algorithm. 
Main resource comes from book 'Python Machine Learning, 3rd version', with additional real exercises. 
Each portion includes codes, charts and explanations, with notes to help my understanding.
I will add more documentation along the learning.
Happy learning!

## Model Evaluation and Hyperparameter Tuning
#### 1) Pipelines
- pipeline, make_pipeline:
    - Chaining preprocessing + estimator into one object. Critical for avoiding data leakage during cross-validation.
- cross_validation:
    - StratifiedKFold: split data by folds, then train and evaluate each of them
    - cross_val_score: split data by folds automatically, then return the score (R^2 for regression, accuracy for classification)
- Learning curves & validation curves
    - learing_curve: performance vs. training set size -> diagnose bias vs. variance
    - LearningCurveDisplay: display the learing curve
    - validation_curve: performance vs. a single hyperparameter -> diagnose under/overfitting for that param
- Hyperparameter tuning:
    - GridSearchSV: search best hyperparameters by all combinations
    - RandomizedSearchCV - better when search space is large
- Evaluation metrics
    - confusion_matrix
    - ConfusionMatrixDisplay
    - precision_score
    - recall_score
    - f1_score
    - roc_curve
    - auc
    - roc_auc_score
    - precision_recall_curve
- Class imbalance
    - class_weight='balanced'
