from sklearn.model_selection import train_test_split
from typing import Sequence, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

def load_data(drop_N_cols=True):
    df = pd.read_csv(Path(".") / ".." / "data" / "train.csv", parse_dates=["tunein", "tuneout"])
    metadata = pd.read_csv(Path(".") / ".." / "data" / "metadata.csv", sep=";")
    df = df.merge(metadata[["content_id", "asset_id"]], right_on="asset_id", left_on="asset_id", how="left")
    if drop_N_cols:
        metadata = metadata[metadata.columns[~(metadata == "N").all()]]
    return df, metadata

def sample_users(df: pd.DataFrame, prop:float=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample users from df.
    
    Parameters
    ----------
      df: Dataframe with column "account_id".
      prop: Proportion between 0 and 1 to sample.
      
    Returns
    -------
      train, test: Train and test sets by account id.
    """
    train_users, test_users = train_test_split(df.account_id.unique(), test_size=prop)
    return df[df.account_id.isin(set(train_users))], df[df.account_id.isin(set(test_users))]

def MAP(y: Sequence[Sequence[int]], yhat: Sequence[Sequence[int]]) -> float:
    """Compute MAP.
    
    Parameters
    ----------
      y: True labels.
      yhat: predictions.
    
    Returns
    -------
      MAP score.
    """
    def AP(y: Sequence[int], yhat: Sequence[int]):
        """Compute AP."""
        n = len(yhat)
        arange = np.arange(1, n+1, dtype=np.int32)
        rel_k = np.in1d(yhat[:n], y)
        tp = np.ones(rel_k.sum(), dtype=np.int32).cumsum()
        denom = arange[rel_k]
        ap = (tp / denom).sum() / len(y)
        return ap
    
    return np.mean([AP(_y, _yhat) for _y, _yhat in zip(y, yhat)])