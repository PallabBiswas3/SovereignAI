"""Measurement-quality review only; never certify a plant fault or healthy state."""
import numpy as np


def assess_wind_measurement_quality(prediction, reference, channel_names):
    x,ref=np.asarray(prediction,dtype=float),np.asarray(reference,dtype=float)
    names=list(channel_names)
    if x.ndim!=2 or ref.ndim!=2 or x.shape[1]!=ref.shape[1] or len(names)!=x.shape[1]:
        raise ValueError('aligned Wind matrices and channel names required')
    bad=~np.isfinite(x).all(axis=1)
    reasons=[]
    if bad.any():reasons.append('nonfinite_prediction_measurements')
    if not np.isfinite(ref).all():
        bad[:]=True;reasons.append('nonfinite_reference_measurements')
    # A genuinely constant reference channel cannot establish a new flatline.
    variable=np.zeros(x.shape[1],dtype=bool)
    for j in range(x.shape[1]):
        finite=ref[np.isfinite(ref[:,j]),j]
        variable[j]=len(finite)>1 and np.std(finite)>1e-12
    runs=np.ones(x.shape[1],dtype=int);flat_channels=set()
    for i in range(1,len(x)):
        equal=np.isfinite(x[i])&np.isfinite(x[i-1])&(np.abs(x[i]-x[i-1])<=1e-12)
        runs=np.where(equal,runs+1,1)
        flat=(runs>=12)&variable
        if flat.any():
            bad[i]=True;flat_channels.update(str(names[j]) for j in np.flatnonzero(flat))
    if flat_channels:reasons.append('flatline_in_reference_variable_channel')
    mask=np.maximum.accumulate(bad)
    indices=np.flatnonzero(mask)
    return {'review_required':bool(len(indices)),'review_mask':mask,
        'first_review_index':int(indices[0]) if len(indices) else None,
        'reasons':reasons,'flatline_channels':sorted(flat_channels),
        'flatline_min_samples':12,'method':'wind_measurement_review_v1',
        'calibrated':False,'interpretation':'check_sensor_or_reference_not_confirmed_equipment_fault'}
