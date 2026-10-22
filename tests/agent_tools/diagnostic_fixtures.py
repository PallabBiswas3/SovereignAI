from copy import deepcopy
from datetime import datetime, timedelta, timezone
from tsdiag.input_contract import VERSION, WORKFLOWS


def registered_turbofan_payload(run_id, source):
    """Fake calibrated artifact for interface tests, not trained-model evidence."""
    import hashlib
    from tsdiag.registry import model_registry
    from tsdiag.input_contract import model_input_profile

    class Model:
        interval_calibration = {"test_fixture_only": True}
        def __call__(self, **kwargs):
            return {"rul_cycles": 10.}
        def predict_with_interval(self, signal, cycles, unit_id=None):
            return {"rul_cycles": 10., "interval": [8., 12.], "calibrated": True,
                    "calibration_dataset": "fake-interface-test", "coverage": .9,
                    "calibration_method": "fixture"}

    example = payload("turbofan")
    example["run_context"].update(run_id=run_id, source=source)
    example["model_refs"] = {"trained_rul_model": "fake-interface-rul-v1"}
    model_registry.register("fake-interface-rul-v1", version="fixture", artifact=Model(),
        checksum=hashlib.sha256(b"fake-interface-rul-v1").hexdigest(),
        metadata={"input_profile": model_input_profile(example)}, replace=True)
    return example


def timestamps(n, start=0):
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [(origin + timedelta(seconds=i + start)).isoformat() for i in range(n)]


def payload(key):
    domain, task, policy, _, _ = WORKFLOWS[key]
    matrix = [[float(i), float(i + 1)] for i in range(32)]
    inputs = {
        "bearing": dict(signal=[float(i % 7) / 10 for i in range(256)], sampling_rate_hz=8000., signal_unit="g", channel_name="acceleration", sensor_location="drive end", bearing_id="specimen-1", shaft_speed_rpm=1800., load_percent=50., fault_frequencies=dict(BPFO=100., BPFI=150., BSF=70., FTF=10.)),
        "process": dict(signal_matrix=matrix, normal_reference=matrix, channel_names=["pressure", "flow"], channel_units=["bar", "m3/h"], sampling_rate_hz=1., reference_id="normal-1", reference_revision="1"),
        "wind_scada": dict(train_matrix=[[float(i), float(i + 1)] for i in range(64)], prediction_matrix=matrix, channel_names=["wind_speed", "active_power"], channel_units=["m/s", "kW"], train_timestamps=timestamps(64), prediction_timestamps=timestamps(32, 100), reference_id="normal-1", reference_revision="1"),
        "battery_diagnosis": dict(cell_voltage=[[3.5, 3.6] for _ in range(32)], cell_temperature=[[25., 26.] for _ in range(32)], cell_ids=["cell-1", "cell-2"], timestamps=timestamps(32, 100), voltage_unit="V", temperature_unit="degC", pack_current=[1.] * 32, current_unit="A"),
        "battery_prognosis": dict(cycle_index=list(range(32)), capacity_ah=[2. - i / 100 for i in range(32)], nominal_capacity_ah=2., eol_capacity_ah=1.4, capacity_unit="Ah", battery_id="specimen-1"),
        "turbofan": dict(signal_matrix=matrix, channel_names=["T2", "T3"], channel_units=["K", "K"], cycle_index=list(range(32)), operating_conditions=[0] * 32, engine_id="specimen-1", regime_map_revision="1", require_calibrated_rul=True),
        "transformer": dict(signal_matrix=matrix, sampling_rate_hz=1000., sensor_positions=["Ia", "Ib"], channel_units=["A", "A"], fundamental_hz=50.),
    }[key]
    return dict(contract_version=VERSION, domain=domain, task=task, policy_ref=policy, inputs=deepcopy(inputs), model_refs={"trained_rul_model": "rul-test-v1"} if key == "turbofan" else {}, run_context={"metadata": dict(asset_id="asset-1", specimen_id="specimen-1", dataset_id="synthetic-contract-tests", dataset_revision="1", observation_timestamp_utc=timestamps(1, 100)[0], acquisition_end_utc=timestamps(1, 200)[0], provenance="synthetic", operating_context="50% load, lab simulation")})
