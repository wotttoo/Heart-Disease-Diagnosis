import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']
COERCE = ['age','trestbps','chol','thalach','oldpeak','ca','thal']

class DataLoader:
    """Handles raw data ingestion and processed split I/O."""

    def __init__(self, data_path: str, splits_dir: str='Data/split'):
        self.data_path = data_path
        self.splits_dir = splits_dir

    # ------------------------------------------------------------------ #
    #  Loading                                                           #
    # ------------------------------------------------------------------ #
    def load_raw(self) -> pd.DataFrame:
        """Load and minimally clean the Cleveland CSV."""
        raw = pd.read_csv(self.data_path)
        raw.columns = COLUMNS

        for  c in COERCE:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')

        raw['target'] = (raw['target'] > 0).astype(int)
        return raw

    def read_split(self, file_path: str) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(Path(file_path))
        X = df.drop(columns='target', axis=1)
        y = df['target']
        print(f'Loaded {Path(file_path).name} - shape X: {X.shape}, y: {y.shape}')
        print(y.value_counts().to_string())
        return X, y
        
    # ------------------------------------------------------------------ #
    #  Saving                                                            #
    # ------------------------------------------------------------------ #
    def save_split(self, splits: dict[str, tuple[pd.DataFrame, pd.Series]]) -> None:
        """Save a dict of {name: (X, y)} pairs to CSV inside splits_dir.

        Example keys: 'raw_train', 'raw_val', 'raw_test', 'fe_train', ...
        """
        Path(self.splits_dir).mkdir(parents=True, exist_ok=True)
        for name, (X, y) in splits.items():
            out = Path(self.splits_dir)/f'{name}.csv'
            pd.concat([X, y.rename('target')], axis=1).to_csv(out, index=False)
            print(f'Saved {out}')

    # ------------------------------------------------------------------ #
    #  Splitting                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def make_splits(
        X: pd.DataFrame,
        y: pd.Series,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series]:
        """Stratified 80 / 10 / 10 split by default.

        Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        temp_size = val_size + test_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_size, stratify=y, random_state=random_state
        )

        test_ratio = test_size/temp_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio, stratify=y_temp, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test