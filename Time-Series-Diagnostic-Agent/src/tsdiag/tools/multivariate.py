from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    return arr


def data_quality_check(signal_matrix, channel_names=None):
    x = _as_2d(signal_matrix)
    names = channel_names or [f"ch{i}" for i in range(x.shape[1])]
    if len(names) != x.shape[1]:
        raise ValueError("channel_names length does not match matrix width")
    stats = {}
    flags = []
    for i, name in enumerate(names):
        col = x[:, i]
        finite = np.isfinite(col)
        missing = float(1 - finite.mean())
        std = float(np.nanstd(col))
        stats[name] = {"missing_fraction": missing, "mean": float(np.nanmean(col)), "std": std}
        if missing > 0:
            flags.append({"channel": name, "flag": "missing_values", "fraction": missing})
        if std < 1e-12:
            flags.append({"channel": name, "flag": "constant_channel"})
    return {"quality_flags": flags, "channel_statistics": stats}


def standardize_against_normal(signal_matrix, normal_reference):
    cur = _as_2d(signal_matrix)
    ref = _as_2d(normal_reference)
    if cur.shape[1] != ref.shape[1]:
        raise ValueError("reference and current channels must align")
    mu = np.nanmean(ref, axis=0)
    scale = np.nanstd(ref, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return {
        "standardized_signal": (cur - mu) / scale,
        "reference_mean": mu,
        "reference_scale": scale,
        "standardized_reference": (ref - mu) / scale,
    }


def pca_monitoring(standardized_signal, normal_reference, variance_target: float = 0.95, alpha: float = 0.99):
    cur = _as_2d(standardized_signal)
    ref = _as_2d(normal_reference)
    if cur.shape[1] != ref.shape[1]:
        raise ValueError("reference/current dimensions differ")
    full = PCA().fit(ref)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_target) + 1)
    n_components = min(max(1, n_components), min(ref.shape))
    pca = PCA(n_components=n_components).fit(ref)
    ref_scores = pca.transform(ref)
    cur_scores = pca.transform(cur)
    ref_recon = pca.inverse_transform(ref_scores)
    cur_recon = pca.inverse_transform(cur_scores)
    ref_spe = np.sum((ref - ref_recon) ** 2, axis=1)
    spe = np.sum((cur - cur_recon) ** 2, axis=1)
    score_mu = ref_scores.mean(axis=0)
    score_cov = np.atleast_2d(np.cov(ref_scores, rowvar=False)) + 1e-9 * np.eye(n_components)
    inv = np.linalg.pinv(score_cov)
    ref_t2 = np.einsum("ij,jk,ik->i", ref_scores - score_mu, inv, ref_scores - score_mu)
    t2 = np.einsum("ij,jk,ik->i", cur_scores - score_mu, inv, cur_scores - score_mu)
    limits = {"spe": float(np.quantile(ref_spe, alpha)), "t2": float(np.quantile(ref_t2, alpha))}
    alarm = (spe > limits["spe"]) | (t2 > limits["t2"])
    return {
        "scores": cur_scores,
        "t2": t2,
        "spe": spe,
        "control_limits": limits,
        "alarm_mask": alarm,
        "pca_state": {
            "components": pca.components_,
            "mean": pca.mean_,
            "explained_variance": pca.explained_variance_,
            "n_components": n_components,
        },
    }


def contribution_analysis(signal_matrix, pca_state, alarm_mask):
    x = _as_2d(signal_matrix)
    mask = np.asarray(alarm_mask, dtype=bool)
    if mask.size != x.shape[0]:
        raise ValueError("alarm_mask length mismatch")
    components = np.asarray(pca_state["components"], dtype=float)
    mean = np.asarray(pca_state["mean"], dtype=float)
    if mask.any():
        active = x[mask]
    else:
        active = x[-min(20, len(x)):]
    scores = (active - mean) @ components.T
    recon = scores @ components + mean
    residual = active - recon
    contrib = np.mean(residual**2, axis=0)
    order = np.argsort(contrib)[::-1]
    total = contrib.sum() + 1e-12
    normalized = contrib / total
    suspects = [int(i) for i in order if normalized[i] >= max(0.1, 1 / (2 * len(normalized)))]
    return {
        "variable_contributions": normalized,
        "suspect_variables": suspects,
        "ranked_indices": order,
    }


def stationarity_analysis(signal_matrix, alpha: float = 0.05):
    x = _as_2d(signal_matrix)
    flags = []
    recommendations = []
    for i in range(x.shape[1]):
        col = x[:, i]
        col = col[np.isfinite(col)]
        if len(col) < 24 or np.std(col) < 1e-12:
            flags.append({"channel": i, "status": "insufficient_or_constant", "adf_p": None, "kpss_p": None})
            recommendations.append({"channel": i, "difference": False})
            continue
        try:
            adf_p = float(adfuller(col, autolag="AIC")[1])
        except Exception:
            adf_p = 1.0
        try:
            kpss_p = float(kpss(col, regression="c", nlags="auto")[1])
        except Exception:
            kpss_p = 0.0
        stationary = adf_p < alpha and kpss_p > alpha
        flags.append({"channel": i, "status": "stationary" if stationary else "nonstationary", "adf_p": adf_p, "kpss_p": kpss_p})
        recommendations.append({"channel": i, "difference": not stationary})
    return {"stationarity_flags": flags, "differencing_recommendations": recommendations}


def granger_causality(signal_matrix, channel_names=None, maxlag: int = 3, alpha: float = 0.05):
    x = _as_2d(signal_matrix)
    names = channel_names or [f"ch{i}" for i in range(x.shape[1])]
    edges = []
    p_values = {}
    for cause in range(x.shape[1]):
        for effect in range(x.shape[1]):
            if cause == effect:
                continue
            pair = np.column_stack([x[:, effect], x[:, cause]])
            key = f"{names[cause]}->{names[effect]}"
            try:
                res = grangercausalitytests(pair, maxlag=maxlag, verbose=False)
                vals = [float(res[lag][0]["ssr_ftest"][1]) for lag in range(1, maxlag + 1)]
                p = min(vals)
                p_values[key] = vals
                if p < alpha:
                    edges.append({"cause": names[cause], "effect": names[effect], "p_value": p})
            except Exception:
                p_values[key] = []
    return {"directed_edges": edges, "p_values": p_values}


def detect_operating_regimes(signal_matrix, operating_regime_labels=None, n_regimes: int = 3, random_state: int = 0):
    x = _as_2d(signal_matrix)
    if operating_regime_labels is not None:
        labels = np.asarray(operating_regime_labels)
        if labels.shape[0] != x.shape[0]:
            raise ValueError("operating_regime_labels length mismatch")
    else:
        z = (x - np.nanmean(x, axis=0)) / (np.nanstd(x, axis=0) + 1e-12)
        labels = KMeans(n_clusters=min(n_regimes, len(x)), random_state=random_state, n_init="auto").fit_predict(np.nan_to_num(z))
    descriptions = {}
    for label in np.unique(labels):
        descriptions[int(label)] = {"samples": int(np.sum(labels == label)), "mean": np.nanmean(x[labels == label], axis=0).tolist()}
    return {"regime_ids": labels, "regime_descriptions": descriptions}


def regime_normalization(signal_matrix, regime_ids, selected_channels=None):
    x = _as_2d(signal_matrix)
    ids = np.asarray(regime_ids)
    if ids.shape[0] != x.shape[0]:
        raise ValueError("regime_ids length mismatch")
    cols = np.arange(x.shape[1]) if selected_channels is None else np.asarray(selected_channels, dtype=int)
    out = x[:, cols].copy()
    state = {}
    for regime in np.unique(ids):
        mask = ids == regime
        mu = np.mean(out[mask], axis=0)
        sd = np.std(out[mask], axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        out[mask] = (out[mask] - mu) / sd
        state[int(regime)] = {"mean": mu, "scale": sd}
    return {"normalized_signal": out, "normalization_state": state}


def residual_analysis(signal_matrix, expected_signal):
    x = _as_2d(signal_matrix)
    expected = _as_2d(expected_signal)
    if x.shape != expected.shape:
        raise ValueError("expected_signal must have same shape as signal_matrix")
    residuals = x - expected
    scale = np.std(residuals, axis=0) + 1e-12
    return {"residuals": residuals, "normalized_residuals": residuals / scale}


def robust_anomaly_detection(normalized_residuals, contamination: float = 0.01, random_state: int = 0):
    x = _as_2d(normalized_residuals)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(x)
    scores = -model.score_samples(x)
    return {"anomaly_scores": scores, "alarm_mask": labels == -1}
