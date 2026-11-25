# model-benchmark-cv

Compares four scikit-learn classifiers (logistic regression, kNN, decision tree, random forest) with 5-fold stratified cross-validation on the built-in breast cancer dataset.

## Run

```bash
pip install -r requirements.txt
python bench.py
pytest
```

## Output (reproduced)

```
$ python3 bench.py
model                  mean acc     std
logistic_regression      0.9789  0.0142
random_forest            0.9667  0.0187
knn                      0.9649  0.0215
decision_tree            0.9262  0.0218

$ pytest -q
3 passed in 7.44s
```

## Limits

- One dataset, default hyperparameters, one CV split seed. Differences of about 1 point are within the reported standard deviation.
