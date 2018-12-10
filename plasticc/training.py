import gc
import os
from datetime import datetime as dt

import numpy as np


np.warnings.filterwarnings('ignore')
gc.enable()


def build_importance_df(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    return importances_


def path_from_cv_score(score, submissions_directory: str=os.path.abspath('../submissions'), suffix: str='') -> str:
    filename = f"subm_{score:.6f}_{dt.now().strftime('%Y-%m-%d-%H-%M')}{suffix}.csv"
    return os.path.join(submissions_directory, filename)
