import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

_GENERATED_NUMS = ['chol_per_age', 'bps_per_age', 'hr_ratio', 'oldpeak_per_age']
_GENERATED_CATS = ['age_bin']

def _add_new_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if set(['chol', 'age']).issubset(X.columns):
        X['chol_per_age'] = X['chol']/X['age']
    if set(['trestbps', 'age']).issubset(X.columns):
        X['bps_per_age'] = X['trestbps']/X['age']
    if set(['thalach', 'age']).issubset(X.columns):
        X['hr_ratio'] = X['thalach']/X['age']
    if set(['oldpeak', 'age']).issubset(X.columns):
        X['oldpeak_per_age'] = X['oldpeak']/X['age']
    if 'age' in X.columns:
        X['age_bin'] = pd.cut(X['age'], bins=5, labels=False).astype('category')
    return X


# --------------------------------------------------------------------------- #
#  Preprocessor                                                               #
# --------------------------------------------------------------------------- #

class HeartDiseasePreprocessor:
    """Builds, fits, and applies sklearn preprocessing pipelines.

    Parameters
    ----------
    mode : 'raw' | 'fe'
        'raw'  — impute + scale/encode original 13 features.
        'fe'   — add engineered features, then impute + scale/encode +
                 variance-threshold filtering.
    """

    def __init__(self, mode: str='raw'):
        if mode not in ('raw', 'fe'):
            raise ValueError("Mode must be'raw' or 'fe'")
        self.mode = mode
        self.pipeline: Pipeline | None = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------ #
    #  Pipeline builders                                                 #
    # ------------------------------------------------------------------ #

    def _build_cat_proc(self) -> Pipeline:
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
    
    def _build_num_proc(self) -> Pipeline:
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

    def _build_column_transformer(
            self, 
            cat_cols: list[str],
            num_cols: list[str]
    ) -> ColumnTransformer:
        return ColumnTransformer([
            ('cat', self._build_cat_proc(), cat_cols),
            ('num', self._build_num_proc(), num_cols)
        ])

    def build(self) -> Pipeline:
        """Construct (but do not fit) the full sklearn Pipeline."""
        if self.mode == 'raw':
            return Pipeline([
            ('preprocess', self._build_column_transformer(CATEGORICAL_COLS, NUMERIC_COLS))
            ])
        else:
            all_cats = CATEGORICAL_COLS + _GENERATED_CATS
            all_nums = NUMERIC_COLS + _GENERATED_NUMS
            return Pipeline([
                ('add_features', FunctionTransformer(_add_new_features, validate=False)),
                ('preprocess', self._build_column_transformer(all_cats, all_nums)),
                ('var_filter', VarianceThreshold())
            ])
    
    # ------------------------------------------------------------------ #
    #  Fit / transform                                                   #
    # ------------------------------------------------------------------ #

    def _extract_feature_names(self) -> list[str]:
        ct = self.pipeline.named_steps['preprocess']
        # Strip 'num__' / 'cat__' transformer prefixes added by ColumnTransformer
        raw_names = ct.get_feature_names_out()
        names = [n.split('__', 1)[1] for n in raw_names]

        if self.mode == 'fe':
            mask = self.pipeline.named_steps['var_filter'].get_support()
            names = [n for n, keep in zip(names, mask) if keep]
        return names

    def fit_transform(
            self, X_train: pd.DataFrame, y_train: pd.Series
        ) -> pd.DataFrame:
        if self.pipeline is None:
            self.pipeline = self.build()
        arr = self.pipeline.fit_transform(X_train, y_train)
        self._feature_names_ = self._extract_feature_names()
        return pd.DataFrame(arr, columns=self._feature_names_, index=X_train.index)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None or not self._feature_names_:
            raise RuntimeError('Call fit_transform before transform.')
        arr = self.pipeline.transform(X)
        return pd.DataFrame(arr, columns=self._feature_names_, index=X.index)
            
    # ------------------------------------------------------------------ #
    #  Mutual-information feature selection                              #
    # ------------------------------------------------------------------ #

    def select_top_k_by_mi(
            self, 
            X_train: pd.DataFrame,
            y_train: pd.Series,
            k: int
    ) -> tuple[list[str], pd.Series]:
        """Rank features by mutual information; return top-k names and full series.

        OHE-encoded columns are treated as discrete; scaled numerics as continuous.
        """
        all_cat_cols = CATEGORICAL_COLS + _GENERATED_CATS
        cat_prefixes = tuple(f'{c}_' for c in all_cat_cols)

        is_discrete = np.array(
            [col.startswith(cat_prefixes) for col in X_train.columns],
            dtype=bool
        )

        mi_scores = mutual_info_classif(
            X_train.values, y_train.values,
            discrete_features=is_discrete,
            random_state=42
        )
        mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
        return list(mi_series.head(k).index), mi_series