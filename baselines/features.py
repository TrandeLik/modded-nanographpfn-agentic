"""GraphLand-style transductive preprocessing and raw, one-hop NFA.

Reference: yandex-research/graphland, dataset.py (transductive loader and
transform defaults); GraphLand paper Appendix B.2 (NFA). Independent NumPy/
SciPy implementation, without requiring PyTorch, PyG, or DGL.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from .data import Dataset


@dataclass
class RawFeatures:
    values: np.ndarray
    numerical: list[int]
    fraction: list[int]
    categorical: list[int]
    names: list[str]


def encode_features(dataset: Dataset) -> RawFeatures:
    """Fit category vocabulary on all nodes, as in GraphLand RL."""
    fractions = dataset.info["fraction_features_names"]
    numerical = [name for name in dataset.info["numerical_features_names"] if name not in fractions]
    categorical = dataset.info["categorical_features_names"]
    numeric_values = dataset.features[numerical + fractions].to_numpy(dtype=np.float32)
    cat_values = np.empty((len(dataset.features), 0), dtype=np.float32)
    cat_names = []
    if categorical:
        encoder = OneHotEncoder(drop="if_binary", sparse_output=False, dtype=np.float32)
        cat_values = encoder.fit_transform(dataset.features[categorical].to_numpy(dtype=np.float32))
        cat_names = encoder.get_feature_names_out(categorical).tolist()
    n_num, n_fraction = len(numerical), len(fractions)
    return RawFeatures(
        np.concatenate([numeric_values, cat_values], axis=1),
        list(range(n_num)), list(range(n_num, n_num + n_fraction)),
        list(range(n_num + n_fraction, n_num + n_fraction + cat_values.shape[1])),
        numerical + fractions + cat_names,
    )


def neighbor_statistics(values: np.ndarray, adjacency: sparse.csr_matrix) -> tuple[np.ndarray, ...]:
    """Per-feature mean/min/max over observed neighbors; empty sets yield NaN.

    Process columns separately so memory does not grow as edges * features.
    np.fmin/fmax ignore NaNs, including when a node has mixed observed/missing
    neighbors. CSR row reductions also handle isolated nodes explicitly.
    """
    means = np.full(values.shape, np.nan, dtype=np.float32)
    minima = np.full_like(means, np.nan)
    maxima = np.full_like(means, np.nan)
    active = np.flatnonzero(np.diff(adjacency.indptr))
    starts = adjacency.indptr[active]
    if not len(active):
        return means, minima, maxima
    for col in range(values.shape[1]):
        x = values[:, col]
        observed = ~np.isnan(x)
        counts = adjacency @ observed.astype(np.float32)
        sums = adjacency @ np.where(observed, x, 0)
        np.divide(sums, counts, out=means[:, col], where=counts > 0)
        neighbors = x[adjacency.indices]
        minima[active, col] = np.fmin.reduceat(neighbors, starts)
        maxima[active, col] = np.fmax.reduceat(neighbors, starts)
    return means, minima, maxima


def add_nfa(raw: RawFeatures, adjacency: sparse.csr_matrix) -> RawFeatures:
    """Append raw numeric neighbor mean/min/max and category frequencies."""
    numeric_columns = sorted(raw.numerical + raw.fraction)
    aggregates = neighbor_statistics(raw.values[:, numeric_columns], adjacency)
    degree = np.diff(adjacency.indptr).astype(np.float32)
    frequencies = np.full((len(degree), len(raw.categorical)), np.nan, dtype=np.float32)
    for col, original in enumerate(raw.categorical):
        sums = adjacency @ raw.values[:, original]
        np.divide(sums, degree, out=frequencies[:, col], where=degree > 0)
    numerical, fraction, names = raw.numerical.copy(), raw.fraction.copy(), raw.names.copy()
    offset = raw.values.shape[1]
    for stat in ("mean", "min", "max"):
        for original in numeric_columns:
            (fraction if original in raw.fraction else numerical).append(offset)
            names.append(f"nfa_{stat}__{raw.names[original]}")
            offset += 1
    for original in raw.categorical:
        fraction.append(offset)
        names.append(f"nfa_frequency__{raw.names[original]}")
        offset += 1
    return RawFeatures(np.concatenate([raw.values, *aggregates, frequencies], axis=1),
                       numerical, fraction, raw.categorical.copy(), names)


def preprocess_features(raw: RawFeatures, base_name: str) -> np.ndarray:
    """Apply GraphLand defaults to all observed node features, never labels.

    NFA numeric statistics inherit their source feature type; categorical
    frequencies are fraction features. All-missing columns are kept and filled
    with zero, an adaptation needed for small/isolated sampled neighborhoods.
    """
    result = raw.values.copy()
    for columns, normalize in ((raw.numerical, True),
                               (raw.fraction, base_name in {"avazu-ctr", "artnet-views"})):
        if not columns:
            continue
        values = result[:, columns]
        nonempty = ~np.isnan(values).all(axis=0)
        if normalize and nonempty.any():
            transformer = QuantileTransformer(n_quantiles=min(1000, len(values)), output_distribution="normal",
                                              subsample=None, random_state=0, copy=False)
            values[:, nonempty] = transformer.fit_transform(values[:, nonempty])
        if np.isnan(values).any():
            imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True, copy=False)
            values = imputer.fit_transform(values)
        result[:, columns] = values
    if not np.isfinite(result).all():
        raise ValueError("Preprocessing produced nonfinite features")
    return np.asarray(result, dtype=np.float32, order="C")
