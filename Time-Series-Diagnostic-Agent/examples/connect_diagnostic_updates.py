"""Build and exercise explicitly labelled synthetic adapters, or serve them.

No production registry defaults are changed. --serve registers these demo-only
artifacts in this process before starting the normal diagnostic API on loopback.
The emitted request files can be used in the authenticated workbench.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
from tsdiag.input_contract import VERSION, WORKFLOWS, model_input_profile
from tsdiag.runtime_updates import (
    BatteryFadeAdapter, BearingCycloAdapter, TransformerSequenceAdapter,
    TurbofanRULAdapter, WindEventAdapter, WindResidualProvider, register_diagnostic_update,
)


def timestamps(n, offset=0):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [(start + timedelta(seconds=offset+i)).isoformat() for i in range(n)]


def envelope(key, inputs):
    domain, task, policy, _, _ = WORKFLOWS[key]
    return {"contract_version": VERSION, "domain": domain, "task": task, "policy_ref": policy,
        "inputs": inputs, "model_refs": {}, "run_context": {"metadata": {
            "asset_id": "SYNTHETIC-DEMO-ASSET", "specimen_id": "demo-evaluation-unit",
            "dataset_id": "synthetic-connection-demo", "dataset_revision": "1",
            "observation_timestamp_utc": timestamps(1, 300)[0], "acquisition_end_utc": timestamps(1, 1000)[0],
            "provenance": "synthetic", "operating_context": "Generated interface demonstration; not plant measurements."}}}


def build_demo():
    from tsdiag.challengers.transformer_sequence import SequenceFeatureClassifier, SequenceFeatureExtractor
    from tsdiag.challengers.turbofan_rul import RULChallenger
    from tsdiag.challengers.battery_prognostics import FadeParticleFilter, fade_curve
    from tsdiag.calibration import EventScorer, EventFARCalibrator, SplitManifest
    from tsdiag.tools.wind_scada import fit_wind_normal_behavior_model
    from tsdiag.uncertainty import ConformalClassifier
    rng = np.random.default_rng(20261104)

    def waveform(label):
        phase = np.arange(100)[:, None]*4*np.pi/100 + np.array([0, -2*np.pi/3, 2*np.pi/3])
        return np.hstack([np.cos(phase), (.5 + 2*label)*np.cos(phase-.3)]) + rng.normal(0, .01, (100, 6))

    labels = np.repeat([0, 1], 40)
    model = SequenceFeatureClassifier(SequenceFeatureExtractor(2), min_per_class=20).fit(np.stack([waveform(y) for y in labels]), labels)
    cy = np.repeat([0, 1], 25)
    conformal = ConformalClassifier(.1, class_conditional=True).fit(model.predict_proba(np.stack([waveform(y) for y in cy])), cy)
    transformer = envelope("transformer", dict(signal_matrix=waveform(1).tolist(), sampling_rate_hz=2500.,
        sensor_positions=["Ua", "Ub", "Uc", "Ia", "Ib", "Ic"], channel_units=["V"]*3+["A"]*3, fundamental_hz=50.))

    windows = rng.normal(size=(80, 4, 2))
    rul = RULChallenger(n_members=1, window=4).fit(windows, np.full(80, 12.), np.repeat(["training-engine-a", "training-engine-b"], 40))
    rul.calibrate(windows[:20], np.full(20, 13.), units=[f"calibration-engine-{i}" for i in range(20)])
    turbofan = envelope("turbofan", dict(signal_matrix=rng.normal(size=(40, 2)).tolist(), channel_names=["T2", "T3"],
        channel_units=["K", "K"], cycle_index=list(range(1, 41)), operating_conditions=[0]*40,
        engine_id="demo-evaluation-unit", regime_map_revision="demo-v1", require_calibrated_rul=True))

    cycles = np.arange(80, dtype=float)
    battery = envelope("battery_prognosis", dict(cycle_index=cycles.tolist(), capacity_ah=fade_curve(cycles, 2., .0025, .006).tolist(),
        nominal_capacity_ah=2., eol_capacity_ah=1.4, capacity_unit="Ah", battery_id="demo-evaluation-unit"))

    fs = 12000.
    t = np.arange(12000)/fs
    signal = rng.normal(size=len(t)) + (1 + .9*np.sin(2*np.pi*85*t))*np.sin(2*np.pi*3000*t)
    bearing = envelope("bearing", dict(signal=signal.tolist(), sampling_rate_hz=fs, signal_unit="g",
        channel_name="acceleration", sensor_location="synthetic drive end", bearing_id="demo-evaluation-unit",
        shaft_speed_rpm=1800., load_percent=50., fault_frequencies=dict(BPFO=85., BPFI=123., BSF=55., FTF=10.)))

    def measurements(n):
        driver = rng.uniform(8, 12, n)
        return np.column_stack([driver, 40 + 2*driver + rng.normal(0, .2, n)])
    training = measurements(400)
    nbm = fit_wind_normal_behavior_model(training, np.zeros(400, dtype=int), target_indices=[1], predictor_indices=[0])
    provider = WindResidualProvider(nbm)
    events = {f"wind-calibration-{i}": provider({"train_matrix": measurements(200), "prediction_matrix": measurements(80)}) for i in range(30)}
    calibrator = EventFARCalibrator(EventScorer(window=3), .1)
    calibrator.fit(events, SplitManifest("event", tuple(events), ("demo-evaluation-unit",)).validate(), n_boot=0)
    wind = envelope("wind_scada", dict(train_matrix=measurements(200).tolist(), prediction_matrix=measurements(80).tolist(),
        channel_names=["wind_speed", "gearbox_temp"], channel_units=["m/s", "degC"],
        train_timestamps=timestamps(200), prediction_timestamps=timestamps(80, 300), reference_id="synthetic-reference", reference_revision="1"))

    return [
        ("transformer", transformer, TransformerSequenceAdapter(model, conformal, {0: "normal", 1: "winding_looseness"})),
        ("turbofan", turbofan, TurbofanRULAdapter(rul)),
        ("battery", battery, BatteryFadeAdapter(FadeParticleFilter(1.4, n_particles=256, seed=4))),
        ("bearing", bearing, BearingCycloAdapter()),
        ("wind", wind, WindEventAdapter(calibrator, provider)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Serve synthetic demonstration artifacts on loopback.")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--output", type=Path, default=PROJECT/"outputs"/"connected-updates-demo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from tsdiag.api import app
    from tsdiag.integrations import diagnose_tool
    summaries = []
    for name, data, adapter in build_demo():
        ref = f"synthetic-demo-{name}-v1"
        data["model_refs"] = {adapter.slot: ref}
        register_diagnostic_update(ref, adapter, version="synthetic-demo-v1",
            input_profile=model_input_profile(data), training_dataset="synthetic-generated-training",
            calibration_dataset="synthetic-generated-calibration", synthetic=True)
        (args.output/f"{name}-request.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        response = diagnose_tool(data)
        (args.output/f"{name}-response.json").write_text(json.dumps(response, indent=2, allow_nan=False), encoding="utf-8")
        if not response["diagnostic_updates"]:
            raise RuntimeError(f"{name} adapter did not execute: {response['abstain_reason']}")
        summaries.append({"domain": name, "model_ref": ref, "decision": response["decision"],
                          "updates": response["diagnostic_updates"]})
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    if args.serve:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
