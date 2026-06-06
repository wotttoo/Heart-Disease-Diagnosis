import os
import random
import warnings

import numpy as np

from src import (
    DataLoader, 
    HeartDiseasePreprocessor,
    EDAVisualizer,
    plot_accuracy_comparison,
    DecisionTreeModel,
    KNNModel,
    NaiveBayesModel,
    EnsembleModel
)

SEED = 42
DATA_PATH = 'Data/raw/cleveland.csv'
SPLITS_DIR = 'Data/split'
ASSETS_DIR = 'outputs/figures'
LABELS     = ["Original Dataset", "Feature Engineering Dataset"]

# ================================================================
# 1. Load & EDA
# ================================================================
loader = DataLoader(DATA_PATH, SPLITS_DIR)
df = loader.load_raw()
print(f'Shape: {df.shape}')
print(df['target'].value_counts().to_string())

viz = EDAVisualizer(df)
viz.plot_num_hist()
viz.plot_cat_target()
viz.plot_scatter_reg()
viz.plot_age_group_mean()
viz.plot_corr_with_age()

# ================================================================
# 2. Train / Val / Test split
# ================================================================
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_val, X_test, y_train, y_val, y_test = loader.make_splits(X, y)
print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

# ================================================================
# 3. Raw preprocessing pipeline
# ================================================================
raw_prep = HeartDiseasePreprocessor(mode='raw')
X_train_raw = raw_prep.fit_transform(X_train, y_train)
X_val_raw = raw_prep.transform(X_val)
X_test_raw = raw_prep.transform(X_test)

# ================================================================
# 4. Feature engineering pipeline
# ================================================================
fe_prep = HeartDiseasePreprocessor(mode='fe')
X_fe_full = fe_prep.fit_transform(X_train, y_train)

top_k, mi_series = fe_prep.select_top_k_by_mi(X_fe_full, y_train, k = X.shape[1])

X_train_fe = X_fe_full[top_k]
X_val_fe = fe_prep.transform(X_val)[top_k]
X_test_fe = fe_prep.transform(X_test)[top_k]

# ================================================================
# 5. Save splits
# ================================================================
loader.save_splits({
    'raw_train': (X_train_raw, y_train),
    'raw_test': (X_test_raw, y_test),
    'raw_val': (X_val_raw, y_val),
    'fe_train': (X_train_fe, y_train),
    'fe_test': (X_test_fe, y_test),
    'fe_val': (X_val_fe, y_val)
})

# ================================================================
# 6. Decision Tree
# ================================================================
print("\n=== Decision Tree ===")

dt_raw = DecisionTreeModel(random_state=SEED).fit(X_train_raw, y_train)
dt_raw_val = dt_raw.evaluate(X_val_raw, y_val, 'Val')
dt_raw_test = dt_raw.evaluate(X_test_raw, y_test, 'Test')

dt_fe = DecisionTreeModel(random_state=SEED).fit(X_train_fe, y_train)
dt_fe_val = dt_fe.evaluate(X_val_fe, y_val, 'Val')
dt_fe_test = dt_fe.evaluate(X_test_fe, y_test, 'Test')

# ================================================================
# 7. KNN
# ================================================================
print("\n=== KNN ===")
knn_raw = KNNModel().fit(X_train_raw, y_train)
knn_raw_val = knn_raw.evaluate(X_val_raw, y_val, 'Val')
knn_raw_test = knn_raw.evaluate(X_test_raw, y_test, 'Test')

knn_fe = KNNModel().fit(X_train_fe, y_train)
knn_fe_val = knn_fe.evaluate(X_val_fe, y_val, 'Val')
knn_fe_test = knn_fe.evaluate(X_test_fe, y_test, 'Test')

# ================================================================
# 8. Naive Bayes
# ================================================================
print("\n=== Naive Bayes ===")

nb_raw = NaiveBayesModel().fit(X_train_raw, y_train)
nb_raw_val = nb_raw.evaluate(X_val_raw, y_val, 'Val')
nb_raw_test = nb_raw.evaluate(X_test_raw, y_test, 'Test')

nb_fe = NaiveBayesModel().fit(X_train_fe, y_train)
nb_fe_val = nb_fe.evaluate(X_val_fe, y_val, 'Val')
nb_fe_test = nb_fe.evaluate(X_test_fe, y_test, 'Test')

# ================================================================
# 9. Ensemble (Stacking)
# ================================================================
print("\n=== Ensemble ===")

ensemble_raw = EnsembleModel(random_state=SEED).fit(X_train_raw, y_train)
ensemble_raw_val = ensemble_raw.evaluate(X_val_raw, y_val, 'Val')
ensemble_raw_test = ensemble_raw.evaluate(X_test_raw, y_test, 'Test')

ensemble_fe = EnsembleModel(random_state=SEED).fit(X_train_fe, y_train)
ensemble_fe_val = ensemble_fe.evaluate(X_val_fe, y_val, 'Val')
ensemble_fe_test = ensemble_fe.evaluate(X_test_fe, y_test, 'Test')

# ================================================================
# 10. Result comparison
# ================================================================
os.makedirs(ASSETS_DIR, exist_ok=True)

for title, val_accs, test_accs, fname in [
    ("Decision Tree", [dt_raw_val,  dt_fe_val],  [dt_raw_test,  dt_fe_test],  "dt"),
    ("KNN",           [knn_raw_val, knn_fe_val], [knn_raw_test, knn_fe_test], "knn"),
    ("Naive Bayes",   [nb_raw_val,  nb_fe_val],  [nb_raw_test,  nb_fe_test],  "nb"),
    ("Ensemble",      [ensemble_raw_val, ensemble_fe_val], [ensemble_raw_test, ensemble_fe_test], "ensemble"),
]:
    plot_accuracy_comparison(
        val_accs  = val_accs,
        test_accs = test_accs,
        labels    = LABELS,
        title     = f"{title} — Val vs Test Accuracy",
        save_path = f"{ASSETS_DIR}/{fname}_accuracy.png",
    )

print(f"\n{'Model':<20} {'Dataset':<25} {'Val':>8} {'Test':>8}")
print("-" * 65)
for model, rvl, rte, fvl, fte in [
    ("Decision Tree", dt_raw_val,  dt_raw_test,  dt_fe_val,  dt_fe_test),
    ("KNN",           knn_raw_val, knn_raw_test, knn_fe_val, knn_fe_test),
    ("Naive Bayes",   nb_raw_val,  nb_raw_test,  nb_fe_val,  nb_fe_test),
    ("Ensemble",      ensemble_raw_val, ensemble_raw_test, ensemble_fe_val, ensemble_fe_test),
]:
    print(f"{model:<20} {'Original':<25} {rvl:>8.4f} {rte:>8.4f}")
    print(f"{'':<20} {'Feature Engineering':<25} {fvl:>8.4f} {fte:>8.4f}")