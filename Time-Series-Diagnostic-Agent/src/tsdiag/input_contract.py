"""Authoritative public diagnostic input contract and form catalogue.

Internal Python runners retain their existing scientific interfaces. HTTP and
JSON tool callers must use this versioned contract; no executable model objects
or decision-threshold overrides are accepted from users.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

VERSION = "industrial-diagnostic-v1"
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Positive = Annotated[Number, Field(gt=0)]
Text = Annotated[str, Field(strict=True, min_length=1, max_length=200, pattern=r".*\S.*")]
Vector = Annotated[list[Number], Field(min_length=8, max_length=200000)]
Matrix = Annotated[list[Annotated[list[Number], Field(min_length=1, max_length=128)]], Field(min_length=8, max_length=200000)]
Names = Annotated[list[Text], Field(min_length=1, max_length=128)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Metadata(StrictModel):
    asset_id: Text
    specimen_id: Text = Field(description="Physical bearing, cell, engine or equipment identifier; used for grouped training splits.")
    dataset_id: Text
    dataset_revision: Text
    observation_timestamp_utc: Text = Field(description="Acquisition start, ISO 8601 with UTC offset; not the upload time.")
    acquisition_end_utc: Text
    provenance: Literal["synthetic", "real", "public_dataset"]
    operating_context: Text = Field(description="Load, operating mode and relevant environmental/collection conditions.")
    source_document_id: Text | None = Field(default=None, description="Authorized observation-log ID when linking diagnostics to document evidence.")

    @model_validator(mode="after")
    def times(self):
        if utc(self.acquisition_end_utc) <= utc(self.observation_timestamp_utc):
            raise ValueError("acquisition_end_utc must be after observation_timestamp_utc")
        return self


def utc(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamps must be ISO 8601 with an explicit UTC offset") from exc
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("timestamps must specify UTC (Z or +00:00)")
    return result.astimezone(timezone.utc)


def matrix_shape(value: list[list[float]], name: str) -> tuple[int, int]:
    width = len(value[0])
    if len(value) * width > 1000000:
        raise ValueError(f"{name} exceeds the one-million-value request limit; submit a bounded analysis window")
    if any(len(row) != width for row in value):
        raise ValueError(f"{name} must be rectangular [rows, channels]")
    return len(value), width


def channel_check(value, names, units, name="signal_matrix"):
    rows, width = matrix_shape(value, name)
    if len(names) != width or len(units) != width or len(set(names)) != width:
        raise ValueError(f"{name}: unique channel names and units must match its columns")
    return rows, width


def aligned_time(values: list[str], rows: int, name: str):
    parsed = [utc(value) for value in values]
    if len(parsed) != rows or any(a >= b for a, b in zip(parsed, parsed[1:])):
        raise ValueError(f"{name} must be strictly increasing and match measurement rows")


def cycles_check(values, rows):
    if len(values) != rows or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("cycle_index must be strictly increasing and match measurement rows")


class Frequencies(StrictModel):
    BPFO: Positive
    BPFI: Positive
    BSF: Positive
    FTF: Positive


class Bearing(StrictModel):
    signal: Annotated[Vector, Field(min_length=256)] = Field(description="Single vibration channel, ordered samples. 256 is an input floor, not proof of adequate frequency resolution.")
    sampling_rate_hz: Positive
    signal_unit: Literal["m/s^2", "g"]
    channel_name: Text
    sensor_location: Text
    bearing_id: Text
    shaft_speed_rpm: Positive
    load_percent: Annotated[Number, Field(ge=0, le=100)]
    fault_frequencies: Frequencies = Field(description="Characteristic frequencies in Hz at the supplied shaft speed; obtain from bearing geometry/manufacturer, do not guess.")

    @model_validator(mode="after")
    def check(self):
        if max(self.fault_frequencies.model_dump().values()) >= self.sampling_rate_hz / 2:
            raise ValueError("fault frequencies must be below Nyquist (sampling_rate_hz / 2)")
        return self


class Process(StrictModel):
    signal_matrix: Matrix = Field(description="Current measurements [samples, channels].")
    normal_reference: Matrix = Field(description="Known healthy reference [samples, same channels], not a copy of the suspect measurements.")
    channel_names: Names
    channel_units: Names
    sampling_rate_hz: Positive
    reference_id: Text
    reference_revision: Text

    @model_validator(mode="after")
    def check(self):
        channel_check(self.signal_matrix, self.channel_names, self.channel_units)
        channel_check(self.normal_reference, self.channel_names, self.channel_units, "normal_reference")
        return self


class Wind(StrictModel):
    train_matrix: Annotated[Matrix, Field(min_length=60)] = Field(description="At least 60 healthy SCADA reference rows; not the held-out prediction rows.")
    prediction_matrix: Matrix
    channel_names: Names
    channel_units: Names
    train_timestamps: list[Text]
    prediction_timestamps: list[Text]
    reference_id: Text
    reference_revision: Text

    @model_validator(mode="after")
    def check(self):
        a, _ = channel_check(self.train_matrix, self.channel_names, self.channel_units, "train_matrix")
        b, _ = channel_check(self.prediction_matrix, self.channel_names, self.channel_units, "prediction_matrix")
        aligned_time(self.train_timestamps, a, "train_timestamps")
        aligned_time(self.prediction_timestamps, b, "prediction_timestamps")
        if utc(self.train_timestamps[-1]) >= utc(self.prediction_timestamps[0]):
            raise ValueError("healthy reference must precede prediction history; overlapping periods are forbidden")
        return self


class BatteryDiagnosis(StrictModel):
    cell_voltage: Matrix = Field(description="[time, cells], volts; columns ordered exactly as cell_ids.")
    cell_temperature: Matrix = Field(description="[time, same cells], degrees Celsius.")
    cell_ids: Names
    timestamps: list[Text]
    voltage_unit: Literal["V"]
    temperature_unit: Literal["degC"]
    pack_current: Vector = Field(description="Amperes, positive charging and negative discharging; one value per row.")
    current_unit: Literal["A"]

    @model_validator(mode="after")
    def check(self):
        rows, width = channel_check(self.cell_voltage, self.cell_ids, ["V"] * len(self.cell_ids), "cell_voltage")
        if matrix_shape(self.cell_temperature, "cell_temperature") != (rows, width) or len(self.pack_current) != rows:
            raise ValueError("voltage, temperature and pack_current must have aligned rows/cells")
        aligned_time(self.timestamps, rows, "timestamps")
        if any(v <= 0 for row in self.cell_voltage for v in row):
            raise ValueError("cell voltage must be positive")
        return self


class BatteryPrognosis(StrictModel):
    cycle_index: Annotated[Vector, Field(min_length=20)]
    capacity_ah: Annotated[Vector, Field(min_length=20)]
    nominal_capacity_ah: Positive
    eol_capacity_ah: Positive
    capacity_unit: Literal["Ah"]
    battery_id: Text

    @model_validator(mode="after")
    def check(self):
        cycles_check(self.cycle_index, len(self.capacity_ah))
        if min(self.cycle_index) < 0 or min(self.capacity_ah) <= 0:
            raise ValueError("cycles must be nonnegative and measured capacities positive")
        if self.eol_capacity_ah >= self.nominal_capacity_ah:
            raise ValueError("eol_capacity_ah must be below nominal_capacity_ah")
        return self


class Turbofan(StrictModel):
    signal_matrix: Matrix
    channel_names: Names
    channel_units: Names
    cycle_index: Vector
    operating_conditions: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=8)] = Field(description="Operating-regime ID per cycle; use a consistent versioned regime map.")
    engine_id: Text
    regime_map_revision: Text
    require_calibrated_rul: Literal[True] = Field(description="Public RUL requests require calibrated model uncertainty; cannot be disabled by clients.")

    @model_validator(mode="after")
    def check(self):
        rows, _ = channel_check(self.signal_matrix, self.channel_names, self.channel_units)
        cycles_check(self.cycle_index, rows)
        if min(self.cycle_index) < 0 or len(self.operating_conditions) != rows:
            raise ValueError("nonnegative cycles and operating_conditions must align with rows")
        return self


class Transformer(StrictModel):
    signal_matrix: Annotated[Matrix, Field(min_length=32)]
    sampling_rate_hz: Positive
    sensor_positions: Names = Field(description="Ordered sensor locations/names, one per column. Model-specific layouts must match the registered artifact.")
    channel_units: Names
    fundamental_hz: Positive = Field(description="Electrical fundamental frequency, Hz.")

    @model_validator(mode="after")
    def check(self):
        channel_check(self.signal_matrix, self.sensor_positions, self.channel_units)
        if self.fundamental_hz >= self.sampling_rate_hz / 2:
            raise ValueError("fundamental_hz must be below Nyquist")
        return self


# Expose only tasks that the current plugins actually execute (not merely plan).
WORKFLOWS = {
    "bearing": ("bearing", "fault_diagnosis", "bearing-policy-v2", Bearing, ()),
    "process": ("process", "root_cause", "process-policy-v3", Process, ("trained_fault_classifier",)),
    "wind_scada": ("wind_scada", "condition_monitoring", "wind-care-event-policy:1.0", Wind, ()),
    "battery_diagnosis": ("battery", "anomaly_localization", "battery-pack-policy-v2", BatteryDiagnosis, ("trained_prognostic_model",)),
    "battery_prognosis": ("battery", "prognosis", "capacity-prognosis-policy-v1", BatteryPrognosis, ()),
    "turbofan": ("turbofan", "remaining_useful_life", "turbofan-policy-v2", Turbofan, ("trained_rul_model",)),
    "transformer": ("transformer", "fault_diagnosis", "transformer-policy-v2", Transformer, ("raw_waveform_model", "trained_image_model")),
}


class Context(StrictModel):
    metadata: Metadata
    run_id: Text | None = None
    source: Text | None = None
    dataset_id: Text | None = None
    protocol_id: Text | None = None
    artifact_checksums: dict[Text, Text] = Field(default_factory=dict)


class Envelope(StrictModel):
    contract_version: Literal[VERSION]
    domain: Text
    task: Text
    policy_ref: Text
    inputs: dict[str, Any]
    model_refs: dict[Text, Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,199}$")]] = Field(default_factory=dict)
    run_context: Context


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Reject malformed public data before any tool, model or retrieval call."""
    try:
        envelope = Envelope.model_validate(payload)
        spec = next((spec for spec in WORKFLOWS.values() if spec[:2] == (envelope.domain, envelope.task)), None)
        if spec is None:
            raise ValueError("unsupported diagnostic domain/task combination")
        _, _, policy, input_model, slots = spec
        if envelope.policy_ref != policy:
            raise ValueError(f"this task requires policy_ref={policy}")
        if set(envelope.model_refs) - set(slots):
            raise ValueError(f"model_refs slots must be chosen from {list(slots)}")
        if envelope.domain == "turbofan" and "trained_rul_model" not in envelope.model_refs:
            raise ValueError("turbofan requires a registered calibrated trained_rul_model reference")
        inputs = input_model.model_validate(envelope.inputs)
        metadata = envelope.run_context.metadata
        duration = (utc(metadata.acquisition_end_utc) - utc(metadata.observation_timestamp_utc)).total_seconds()
        samples = len(inputs.signal) if isinstance(inputs, Bearing) else len(inputs.signal_matrix) if isinstance(inputs, (Process, Transformer)) else None
        if samples and duration < (samples - 1) / inputs.sampling_rate_hz:
            raise ValueError("acquisition interval is shorter than the signal duration at sampling_rate_hz")
        if isinstance(inputs, Bearing) and inputs.bearing_id != metadata.specimen_id:
            raise ValueError("bearing_id must match specimen_id")
        if isinstance(inputs, BatteryPrognosis) and inputs.battery_id != metadata.specimen_id:
            raise ValueError("battery_id must match specimen_id")
        if isinstance(inputs, Turbofan) and inputs.engine_id != metadata.specimen_id:
            raise ValueError("engine_id must match specimen_id")
        for field in ("timestamps", "prediction_timestamps"):
            values = getattr(inputs, field, None)
            if values and (utc(values[0]) < utc(metadata.observation_timestamp_utc) or utc(values[-1]) > utc(metadata.acquisition_end_utc)):
                raise ValueError(f"{field} must lie within the declared acquisition interval")
        return envelope.model_dump(mode="json") | {"inputs": inputs.model_dump(mode="json")}
    except ValidationError as exc:
        errors = "; ".join(f"{'.'.join(map(str, row['loc']))}: {row['msg']}" for row in exc.errors(include_input=False, include_url=False))
        raise ValueError(errors) from exc


def catalogue() -> dict[str, Any]:
    return {
        "contract_version": VERSION,
        "metadata_schema": Metadata.model_json_schema(),
        "workflows": [
            {"id": key, "domain": domain, "task": task, "policy_ref": policy,
             "input_schema": model.model_json_schema(), "model_slots": list(slots),
             "required_model_slots": ["trained_rul_model"] if domain == "turbofan" else []}
            for key, (domain, task, policy, model, slots) in WORKFLOWS.items()
        ],
    }


def model_input_profile(payload: dict[str, Any]) -> dict[str, Any]:
    """Training/inference compatibility key; labels never belong in inputs.

    Artifacts must additionally version their windowing and preprocessing. This
    profile prevents silently reusing a model with different channel order,
    units, sampling rate or task; it does not certify the training data.
    """
    data = validate_payload(payload)
    inputs = data["inputs"]
    return {
        "contract_version": VERSION, "domain": data["domain"], "task": data["task"],
        **{key: inputs[key] for key in (
            "channel_names", "channel_units", "sensor_positions", "sampling_rate_hz",
            "signal_unit", "channel_name", "voltage_unit", "temperature_unit", "current_unit",
            "capacity_unit", "regime_map_revision",
        ) if key in inputs},
    }
