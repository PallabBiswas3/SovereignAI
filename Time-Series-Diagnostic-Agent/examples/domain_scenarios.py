"""Synthetic normal and fault inputs for every public pipeline domain."""

import numpy as np


def build_scenarios(seed: int = 20):
    rng = np.random.default_rng(seed)
    fs = 8000.0
    t = np.arange(0, 1.0, 1/fs)
    bearing_fault = (1 + .9*np.sin(2*np.pi*80*t))*np.sin(2*np.pi*1800*t)
    bearing_meta = {"sampling_rate_hz":fs,"fault_frequencies":{"BPFO":80.,"BPFI":125.,"BSF":55.,"FTF":10.}}

    process_ref = rng.normal(size=(180,3)); process_fault = rng.normal(size=(100,3)); process_fault[40:,0] += 4
    process_common = {"normal_reference":process_ref,"channel_names":["pressure","flow","level"],"sampling_rate_hz":1.,"maxlag":1}

    scada_ref=rng.normal(scale=.2,size=(100,3)); scada_fault=rng.normal(scale=.2,size=(100,3)); scada_fault[40:,1]+=2
    scada_common={"train_matrix":scada_ref,"channel_names":["power","gearbox_temp","wind"],"prediction_timestamps":np.arange(100)}

    voltage=3.7+rng.normal(scale=.003,size=(80,5)); temperature=30+rng.normal(scale=.1,size=(80,5)); bad_voltage=voltage.copy(); bad_voltage[:,3]-=.12
    battery_common={"cell_temperature":temperature,"cell_ids":[f"cell_{i}" for i in range(5)],"timestamps":np.arange(80)}

    cycles=np.arange(1,121); turbo_normal=rng.normal(scale=.2,size=(120,3)); turbo_fault=turbo_normal.copy(); turbo_fault[:,0]+=np.linspace(0,4,120)
    turbo_common={"channel_names":["temperature","pressure","vibration"],"cycle_index":cycles}

    tf_t=np.arange(4096)/4000.; smooth=np.column_stack([np.sin(2*np.pi*300*tf_t),.8*np.sin(2*np.pi*300*tf_t)])
    impulse=np.zeros(4096); impulse[::100]=8; impacted=smooth+impulse[:,None]
    transformer_common={"sampling_rate_hz":4000.,"sensor_positions":["tank_left","tank_right"]}
    classifier=lambda image:{"label":"winding_looseness","confidence":.82,"probabilities":{"winding_looseness":.82}}

    return {
      "bearing": {"normal":{**bearing_meta,"signal":rng.normal(scale=.1,size=len(t)),"minimum_confidence":.99},"fault":{**bearing_meta,"signal":bearing_fault}},
      "process": {"normal":{**process_common,"signal_matrix":process_ref[:100]},"fault":{**process_common,"signal_matrix":process_fault}},
      "wind_scada": {"normal":{**scada_common,"prediction_matrix":rng.normal(scale=.2,size=(100,3))},"fault":{**scada_common,"prediction_matrix":scada_fault}},
      "battery": {"normal":{**battery_common,"cell_voltage":voltage},"fault":{**battery_common,"cell_voltage":bad_voltage}},
      "turbofan": {"normal":{**turbo_common,"signal_matrix":turbo_normal},"fault":{**turbo_common,"signal_matrix":turbo_fault}},
      "transformer": {"normal":{**transformer_common,"signal_matrix":smooth},"fault":{**transformer_common,"signal_matrix":impacted,"trained_image_model":classifier}},
    }
