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

def sample_users(df: pd.DataFrame, prop:float=0.2, seed:int=117) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample users from df.
    
    Parameters
    ----------
      df: Dataframe with column "account_id".
      prop: Proportion between 0 and 1 to sample.
      
    Returns
    -------
      train, test: Train and test sets by account id.
    """
    train_users, test_users = train_test_split(df.account_id.unique(), test_size=prop, random_state=117)
    return df[df.account_id.isin(set(train_users))], df[df.account_id.isin(set(test_users))]

# source: https://github.com/raminqaf/Metrics/blob/fix/average-percision-calculation/Python/ml_metrics/average_precision.py
def apk(actual: list, predicted: list, k=10) -> float:
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted
    predicted : list
             A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : float
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    sum_precision = 0.0
    num_hits = 0.0

    for i, prediction in enumerate(predicted):
        if prediction in actual[:k] and prediction not in predicted[:i]:
            num_hits += 1.0
            precision_at_i = num_hits / (i + 1.0)
            sum_precision += precision_at_i

    if num_hits == 0.0:
        return 0.0

    return sum_precision / num_hits


def mapk(actual: list, predicted: list, k=10) -> float:
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


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
    return mapk(y, yhat, 20)