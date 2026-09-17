from __future__ import annotations

import numpy as np
import inspect

from ..execution import DomainToolRegistry
from ..tools.change import change_point_detection, cross_sensor_relationships, rolling_statistics
from ..tools.wavelet import correlation_sensor_weighting, cross_correlation_analysis, multisensor_fusion, wavelet_denoising


def matrix(value, name="signal_matrix", minimum=8):
    x = np.asarray(value, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) < minimum or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a finite [samples, channels] matrix with at least {minimum} samples")
    return x


def names(value, width, label):
    out = [str(row) for row in value]
    if len(out) != width:
        raise ValueError(f"{label} length must match channels")
    return out


# Wind SCADA
def scada_quality(s):
    x = matrix(s["signal_matrix"]); ts = np.asarray(s["timestamps"])
    n = names(s["channel_names"], x.shape[1], "channel_names")
    if ts.ndim != 1 or len(ts) != len(x): raise ValueError("timestamps must align with samples")
    return {"signal_matrix": x, "timestamps": ts, "channel_names": n, "quality_flags": []}


def scada_regimes(s):
    labels = s.get("operating_regime_labels")
    ids = np.asarray(labels, dtype=int) if labels is not None else np.zeros(len(s["signal_matrix"]), dtype=int)
    if len(ids) != len(s["signal_matrix"]): raise ValueError("operating_regime_labels must align with samples")
    return {"regime_ids": ids, "regime_descriptions": {str(i): int(np.sum(ids == i)) for i in np.unique(ids)}}


def scada_normal(s):
    x = s["signal_matrix"]; model = s.get("physics_model")
    ref = matrix(s["normal_reference"]) if s.get("normal_reference") is not None else x[:max(8, len(x)//4)]
    if ref.shape[1] != x.shape[1]: raise ValueError("normal_reference channels must match")
    center = np.median(ref, axis=0); scale = np.median(np.abs(ref-center), axis=0)*1.4826
    scale = np.where(scale > 1e-9, scale, np.where(ref.std(0)>1e-9, ref.std(0), 1.0))
    if callable(model):
        raw = model(signal_matrix=x, channel_names=s["channel_names"], timestamps=s["timestamps"])
        expected = np.asarray(raw.get("expected_signal") if isinstance(raw, dict) else raw, dtype=float)
        if expected.shape != x.shape: raise ValueError("physics_model output must match signal_matrix")
    else: expected = np.broadcast_to(center, x.shape).copy()
    return {"expected_signal": expected, "reference_scale": scale, "model_uncertainty": float(np.mean(scale))}


def scada_residual(s):
    residual = s["signal_matrix"] - s["expected_signal"]
    return {"residuals": residual, "normalized_residuals": residual/s["reference_scale"]}


def scada_rolling(s): return rolling_statistics(s["normalized_residuals"], s["timestamps"], window=min(max(3,len(s["signal_matrix"])//10),20))
def scada_relationships(s): return cross_sensor_relationships(s["signal_matrix"], s["channel_names"])


def scada_anomaly(s):
    scores=np.max(np.abs(s["normalized_residuals"]),axis=1); mask=scores>float(s.get("z_threshold",4.0))
    return {"anomaly_scores":scores,"alarm_mask":mask,"alarm_fraction":float(np.mean(mask))}


def scada_changes(s): return change_point_detection(s["normalized_residuals"],s["timestamps"],min_size=max(3,min(20,len(s["signal_matrix"])//4)))


def scada_physics(s):
    return {"verification_findings":{"status":"SUPPORTED" if np.any(s["alarm_mask"]) else "INSUFFICIENT","physics_model_used":callable(s.get("physics_model"))}}


def scada_decision(s):
    channel_scores=np.quantile(np.abs(s["normalized_residuals"]),.95,axis=0); top=int(np.argmax(channel_scores))
    abnormal=s["alarm_fraction"]>=float(s.get("minimum_alarm_fraction",.05)); conf=float(np.clip(max(s["alarm_fraction"]*4,channel_scores[top]/8),.05,.95))
    return {"abnormal":abnormal,"confidence":conf,"top_channel":s["channel_names"][top],"channel_scores":channel_scores}


# Battery
def battery_quality(s):
    v=matrix(s["cell_voltage"],"cell_voltage"); t=matrix(s["cell_temperature"],"cell_temperature")
    if v.shape != t.shape: raise ValueError("cell voltage and temperature shapes must match")
    ids=names(s["cell_ids"],v.shape[1],"cell_ids"); ts=np.asarray(s["timestamps"])
    if len(ts)!=len(v): raise ValueError("timestamps must align with samples")
    return {"cell_voltage":v,"cell_temperature":t,"cell_ids":ids,"timestamps":ts,"quality_flags":[]}


def battery_states(s):
    current=s.get("pack_current"); labels=np.zeros(len(s["cell_voltage"]),dtype=int) if current is None else np.sign(np.asarray(current)).astype(int)
    if len(labels)!=len(s["cell_voltage"]): raise ValueError("pack_current must align with samples")
    return {"state_labels":labels,"state_intervals":[]}


def _cell_deviation(x):
    delta=x-np.median(x,axis=1,keepdims=True); offset=np.median(delta,axis=0); centered=delta-offset
    noise_rows=np.median(np.abs(centered),axis=0)*1.4826
    noise=float(np.median(noise_rows[noise_rows>1e-9])) if np.any(noise_rows>1e-9) else float(np.std(centered)+1e-6)
    return delta,np.abs(offset)/max(noise,1e-6)


def battery_deviation(s):
    dv,sv=_cell_deviation(s["cell_voltage"]); dt,st=_cell_deviation(s["cell_temperature"])
    return {"delta_voltage":dv,"delta_temperature":dt,"imbalance_scores":np.maximum(sv,st)}


def battery_temporal(s):
    return {"dv_dt":np.gradient(s["cell_voltage"],axis=0),"dt_dt":np.gradient(s["cell_temperature"],axis=0),"temporal_features":{"voltage_slope":np.mean(np.gradient(s["cell_voltage"],axis=0),axis=0)}}


def battery_spatial(s):
    score=s["imbalance_scores"]; return {"cell_similarity":np.corrcoef(s["cell_voltage"],rowvar=False),"spatial_anomaly_scores":score}


def battery_windows(s):
    n=len(s["timestamps"]); width=max(8,min(n,int(s.get("window_length",max(8,n//4)))))
    return {"windows":[(i,min(i+width,n)) for i in range(0,n,width)],"window_metadata":{"length":width}}


def battery_model(s):
    model=s.get("trained_prognostic_model"); prediction={}
    if callable(model):
        raw=model(cell_voltage=s["cell_voltage"],cell_temperature=s["cell_temperature"],timestamps=s["timestamps"])
        prediction=raw if isinstance(raw,dict) else {"risk":float(raw)}
    return {"cell_fault_scores":s["spatial_anomaly_scores"],"pack_fault_score":float(np.max(s["spatial_anomaly_scores"])),"latent_health_state":prediction}


def battery_localize(s):
    scores=np.asarray(s["cell_fault_scores"]); order=np.argsort(scores)[::-1]
    return {"ranked_cells":[s["cell_ids"][i] for i in order],"top_cell":s["cell_ids"][order[0]],"top_score":float(scores[order[0]])}


def battery_prognosis(s):
    pred=s["latent_health_state"]; conf=float(np.clip(s["top_score"]/(float(s.get("cell_anomaly_threshold",3.5))*1.5),.05,.95))
    risk=float(np.clip(pred.get("risk",np.max(np.maximum(s["cell_temperature"]-50,0))/20+conf*.5),0,1))
    return {"failure_probability_by_horizon":risk,"prognosis_uncertainty":float(pred.get("uncertainty",1-conf)),"confidence":conf}


def battery_decision(s):
    abnormal=s["top_score"]>=float(s.get("cell_anomaly_threshold",3.5))
    return {"abnormal":abnormal,"condition":"cell_imbalance" if abnormal else "normal"}


# Turbofan
def turbo_screen(s):
    x=matrix(s["signal_matrix"]); cycles=np.asarray(s["cycle_index"],dtype=float); n=names(s["channel_names"],x.shape[1],"channel_names")
    if len(cycles)!=len(x) or np.any(np.diff(cycles)<=0): raise ValueError("cycle_index must be strictly increasing and aligned")
    z=(x-np.median(x,axis=0))/(np.std(x,axis=0)+1e-9); span=max(float(np.ptp(cycles)),1)
    slopes=np.array([np.polyfit(cycles,z[:,j],1)[0]*span for j in range(x.shape[1])]); selected=np.flatnonzero(np.abs(slopes)>=max(.1,np.quantile(np.abs(slopes),.5)))
    if not len(selected): selected=np.array([int(np.argmax(np.abs(slopes)))])
    return {"signal_matrix":x,"cycle_index":cycles,"channel_names":n,"standardized_signal":z,"sensor_scores":np.abs(slopes),"selected_indices":selected,"selected_channels":[n[i] for i in selected],"sensor_slopes":slopes}


def turbo_regimes(s):
    ids=np.asarray(s.get("operating_conditions",np.zeros(len(s["signal_matrix"]))))
    if ids.ndim>1: ids=np.argmax(ids,axis=1)
    return {"regime_ids":ids.astype(int),"regime_statistics":{str(i):int(np.sum(ids==i)) for i in np.unique(ids)}}


def turbo_normalize(s): return {"normalized_signal":s["standardized_signal"],"normalization_state":{"method":"robust_global"}}


def turbo_smooth(s):
    x=s["normalized_signal"]; width=max(3,min(11,len(x)//5)); kernel=np.ones(width)/width
    trend=np.column_stack([np.convolve(x[:,j],kernel,mode="same") for j in range(x.shape[1])])
    return {"trend_signal":trend,"residual_signal":x-trend}


def turbo_windows(s):
    x=s["normalized_signal"]; length=min(len(x),int(s.get("window_length",30))); stride=int(s.get("stride",max(1,length//2)))
    windows=np.stack([x[i:i+length] for i in range(0,len(x)-length+1,stride)])
    return {"sequence_windows":windows,"window_end_cycles":[float(s["cycle_index"][i+length-1]) for i in range(0,len(x)-length+1,stride)]}


def turbo_health(s):
    idx=s["selected_indices"]; direction=np.sign(s["sensor_slopes"][idx]); h=np.mean(s["normalized_signal"][:,idx]*direction,axis=1); h-=h[0]
    return {"health_index":h,"health_contributors":s["selected_channels"]}


def turbo_rul(s):
    h=s["health_index"]; cycles=s["cycle_index"]; slope,intercept=np.polyfit(cycles,h,1); model=s.get("trained_rul_model")
    if callable(model):
        raw=model(signal_matrix=s["signal_matrix"],cycle_index=cycles,health_index=h); rul=float(raw["rul_cycles"] if isinstance(raw,dict) else raw); method="trained"
    else: rul=max(0,float((float(s.get("failure_threshold",3))-intercept)/slope-cycles[-1])) if slope>1e-9 else None; method="linear"
    residual=h-(slope*cycles+intercept); return {"rul_cycles":rul,"rul_method":method,"health_slope":float(slope),"fit_error":float(np.sqrt(np.mean(residual**2)))}


def turbo_uncertainty(s):
    conf=float(np.clip(1/(1+s["fit_error"]),.1,.9)); rul=s["rul_cycles"]
    return {"rul_interval":None if rul is None else [max(0,rul*(2-conf)),rul*(1+1-conf)],"uncertainty_score":1-conf,"confidence":conf}


def turbo_explain(s): return {"critical_sensors":s["health_contributors"],"explanation":"Sensors with the strongest normalized monotonic trends."}


def turbo_decision(s):
    current=float(np.mean(s["health_index"][-max(3,len(s["health_index"])//10):])); abnormal=current>=float(s.get("degradation_threshold",1)) and s["health_slope"]*max(float(np.ptp(s["cycle_index"])),1)>=1
    return {"abnormal":abnormal,"health_state":"degraded" if abnormal else "stable","current_health":current}


# Transformer
def transformer_sync(s):
    x=matrix(s["signal_matrix"],minimum=32); fs=float(s["sampling_rate_hz"]); pos=names(s["sensor_positions"],x.shape[1],"sensor_positions")
    if fs<=0 or not np.isfinite(fs): raise ValueError("sampling_rate_hz must be positive")
    return {"signal_matrix":x,"sampling_rate_hz":fs,"sensor_positions":pos,"quality_flags":[],"sync_error":0.0,"channel_statistics":{"std":x.std(0)}}


def transformer_denoise(s): return wavelet_denoising(s["signal_matrix"],s.get("wavelet","db4"),s.get("level"),s.get("threshold_rule","soft"))
def transformer_corr(s): return cross_correlation_analysis(s["denoised_signal_matrix"])
def transformer_weights(s): return correlation_sensor_weighting(s["correlation_energy"])
def transformer_fuse(s): return multisensor_fusion(s["denoised_signal_matrix"],s["sensor_weights"])


def transformer_envelope(s):
    x=s["fused_waveform"]-np.mean(s["fused_waveform"]); v=float(np.mean(x*x)); k=float(np.mean(x**4)/(v*v+1e-12))
    return {"envelope_spectrum":np.abs(np.fft.rfft(np.abs(x))),"harmonic_structure":{"kurtosis":k,"anomaly_score":float(np.clip(max(k-3,0)/7,0,1))}}


def transformer_spectral(s):
    x=s["fused_waveform"]-np.mean(s["fused_waveform"])
    # Preserve time resolution for short fixed windows such as SGAH's 100-sample
    # events. The old width=len(x) produced exactly one FFT frame and therefore
    # an all-zero temporal-difference representation.
    width=min(256,max(16,len(x)//2)); hop=max(1,width//2)
    rows=[x[i:i+width] for i in range(0,len(x)-width+1,hop)]
    spec=np.stack([np.abs(np.fft.rfft(r*np.hanning(width))) for r in rows],axis=1)
    corr=np.abs(np.diff(spec,axis=1,prepend=spec[:,:1]))
    return {"spectral_correlation_map":corr,"cyclic_frequency_axis":np.arange(corr.shape[1]),"carrier_frequency_axis":np.fft.rfftfreq(width,1/s["sampling_rate_hz"])}


def transformer_representation(s):
    image=np.log1p(s["spectral_correlation_map"]); image/=max(float(np.max(image)),1e-12)
    return {"feature_image":image,"representation_metadata":{"shape":list(image.shape)}}


def transformer_classify(s):
    raw_model=s.get("raw_waveform_model")
    image_model=s.get("trained_image_model")
    if callable(raw_model):
        raw=raw_model(s["signal_matrix"]); family=getattr(raw_model,"model_family","AD-TFM-AT")
    elif callable(image_model):
        raw=image_model(s["feature_image"]); family=getattr(image_model,"model_family","spectral_image")
    else:
        return {"fault_probabilities":{},"predicted_fault":None,"classifier_model_family":None}
    out=raw if isinstance(raw,dict) else {"label":str(raw)}
    return {
      "fault_probabilities":out.get("probabilities",{}),"predicted_fault":out.get("label"),
      "predicted_class":out.get("predicted_class",out.get("label")),
      "model_confidence":float(out.get("confidence",.5)),
      "classifier_model_family":out.get("model_family",family),
    }


def transformer_decision(s):
    score=s["harmonic_structure"]["anomaly_score"]; abnormal=score>=float(s.get("anomaly_threshold",.25)); label=s["predicted_fault"]
    return {"abnormal":abnormal,"fault_label":label,"confidence":float(np.clip(s.get("model_confidence",max(.1,score)),0,1)),"abstain_reason":"Abnormal evidence requires a trained fault classifier." if abnormal and not label else None}


def default_domain_tool_registry():
    r=DomainToolRegistry()
    groups={
      "wind_scada":[scada_quality,scada_regimes,scada_normal,scada_residual,scada_rolling,scada_relationships,scada_anomaly,scada_changes,scada_physics,scada_decision],
      "battery":[battery_quality,battery_states,battery_deviation,battery_temporal,battery_spatial,battery_windows,battery_model,battery_localize,battery_prognosis,battery_decision],
      "turbofan":[turbo_screen,turbo_regimes,turbo_normalize,turbo_smooth,turbo_windows,turbo_health,turbo_rul,turbo_uncertainty,turbo_explain,turbo_decision],
      "transformer":[transformer_sync,transformer_denoise,transformer_corr,transformer_weights,transformer_fuse,transformer_envelope,transformer_spectral,transformer_representation,transformer_classify,transformer_decision],
    }
    from . import DOMAIN_PACKS
    for domain, funcs in groups.items():
        for contract, fn in zip(DOMAIN_PACKS[domain].tools, funcs): r.register(domain,contract.name,fn)
    from ..tools.registry import default_tool_registry
    from ..tools.bearing import bearing_evidence_fusion, bearing_frequency_match
    generic = default_tool_registry()
    extras = {"bearing_frequency_match": bearing_frequency_match, "bearing_evidence_fusion": bearing_evidence_fusion}

    def adapt(fn, aliases=None):
        aliases = aliases or {}
        signature = inspect.signature(fn)
        def run(state):
            kwargs = {}
            for name, parameter in signature.parameters.items():
                key = name if name in state else aliases.get(name, name)
                if key in state:
                    kwargs[name] = state[key]
                elif parameter.default is inspect.Parameter.empty:
                    raise ValueError(f"missing tool input {key!r}")
            output = fn(**kwargs)
            return output if isinstance(output, dict) else {"result": output}
        return run

    aliases = {
        "bandpass_filter": {"band_hz": "recommended_band_hz"},
        "bearing_evidence_fusion": {"time_features": "time_domain_features"},
        "standardize_against_normal": {"normal_reference": "normal_reference"},
    }
    for domain in ("bearing", "process"):
        for contract in DOMAIN_PACKS[domain].tools:
            fn = extras.get(contract.name) or generic.get(contract.name)
            if fn is not None:
                r.register(domain, contract.name, adapt(fn, aliases.get(contract.name)))
    return r
