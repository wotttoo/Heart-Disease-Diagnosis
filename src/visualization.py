import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from .preprocessing import CATEGORICAL_COLS, NUMERIC_COLS

class EDAVisualizer:
    """Exploratory plots for the Heart Disease dataset.

    Parameters
    ----------
    df : DataFrame that includes the target column.
    target_col : Name of the binary target column.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'target'):
        self.df = df
        self.target_col = target_col
        self._colors = ['tab:blue' if t == 0 else 'tab:red' for t in df[target_col]]


    # ------------------------------------------------------------------ #
    #  Distribution plots                                                #
    # ------------------------------------------------------------------ #

    def plot_num_hist(self) -> None:
        """Histograms for numeric features."""
        self.df[NUMERIC_COLS].hist(figsize=(10, 8))
        plt.tight_layout()
        plt.savefig('outputs/figures/numericals_hist.png')
        plt.show()
    
    def plot_cat_target(self):
        """Stacked bar chart of target distribution per categorical feature."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()

        for i, cat in enumerate(CATEGORICAL_COLS):
            self.df.groupby([cat, 'target']).size().unstack().plot(
                kind='bar', stacked=True, ax=axes[i]
            )
            axes[i].set_title(f'Distribution of target by {cat}')
            axes[i].set_xlabel(cat)
            axes[i].set_ylabel('count')
            axes[i].tick_params(axis='x', rotation=0)
        
        for j in range(len(CATEGORICAL_COLS), 9):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig('outputs/figures/categorical_target_dist.png')
        plt.show()

    # ------------------------------------------------------------------ #
    #  Scatter + regression                                              #
    # ------------------------------------------------------------------ #

    def plot_scatter_reg(self, target_cols: list[str] | None = None) -> None:
        cols_to_plot = target_cols or ['trestbps', 'chol','thalach', 'oldpeak']

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()

        for i, col in enumerate(cols_to_plot):
            # Drop NaNs for fitting
            mask = self.df['age'].notna() & self.df[col].notna()
            x = self.df['age'][mask].values.reshape(-1, 1)
            yy = self.df[col][mask].values

            if len(x) > 1:
                model = LinearRegression()
                model.fit(x, yy)
                # Create line
                x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
                y_pred = model.predict(x_line)
            else:
                x_line, y_pred = None, None

            axes[i].scatter(self.df['age'], self.df[col], color=self._colors, alpha=0.7)
            axes[i].set_title(f'Age vs. {col} (+ regression line)')
            axes[i].set_xlabel(f'Age')
            axes[i].set_ylabel(col)

            if x_line is not None:
                axes[i].plot(x_line.ravel(), y_pred)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/scatter_numeric_vs_age_regression.png')
        plt.show()

    # ------------------------------------------------------------------ #
    #  Age-group aggregation                                             #
    # ------------------------------------------------------------------ #

    def plot_age_group_mean(self, target_cols: list[str] | None = None) -> None:
        """Line chart of per-age-bin averages for selected numeric features."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()

        age_bins = pd.cut(self.df['age'], bins=5)
        target_cols = target_cols or ['trestbps', 'chol','thalach', 'oldpeak']

        for i, col in enumerate(target_cols):
            avg_vals = self.df.groupby(age_bins)[col].mean()
            axes[i].plot(range(len(avg_vals)), avg_vals, marker='o', label=f'Average {col}')
            axes[i].set_xticks(range(len(avg_vals)))
            axes[i].set_xticklabels([str(j) for j in avg_vals.index], rotation=45, ha='right')
            axes[i].set_title(f'{col} average per age group')
            axes[i].set_xlabel('Age group')
            axes[i].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/numeric_mean_by_age_mean.png')
        plt.show()

    def plot_corr_with_age(self, target_cols: list[str] | None = None) -> None:
        """Bar chart of Pearson correlation with age."""
        target_cols = target_cols or ['age', 'chol', 'trestbps', 'thalach', 'oldpeak']
        corr = self.df[target_cols].corr(numeric_only=True)['age'].drop('age')

        plt.figure()
        corr.plot(kind='bar')
        plt.title('Correlation of features with age')
        plt.ylabel('Correlations with age')
        plt.tight_layout()
        plt.savefig('outputs/figures/features_corr_with_age.png')
        plt.show()
    
