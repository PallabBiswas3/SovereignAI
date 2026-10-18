"""Conservative reference compatibility and unique-root identifiability checks.

These are retrospective diagnostic checks, not online alarm thresholds or a
calibrated causal model. Declared regime equality does not prove data authenticity.
"""
from typing import Any, Mapping

import numpy as np


def assess_process_reference(inputs: Mapping[str, Any]) -> dict[str, Any]:
    current = inputs.get('current_regime_id')
    reference = inputs.get('reference_regime_id')
    if (current is None) != (reference is None):
        raise ValueError('current_regime_id and reference_regime_id must be supplied together')
    if current is not None and (not isinstance(current,str) or not current.strip() or not isinstance(reference,str) or not reference.strip()):
        raise ValueError('regime identifiers must be nonempty strings')
    assessment = {'regime_comparison': 'not_declared' if current is None else 'declared_match',
                  'reason': None, 'competing_channels': [], 'timing_mode': 'retrospective_reference_based'}
    if current is not None and current != reference:
        return {**assessment, 'regime_comparison': 'declared_mismatch', 'reason': 'reference_regime_mismatch'}
    return assessment


def assess_root_identifiability(inputs: Mapping[str, Any]) -> dict[str, Any]:
    assessment = assess_process_reference(inputs)
    if assessment['reason']:
        return assessment
    x = np.asarray(inputs['signal_matrix'], dtype=float)
    ref = np.asarray(inputs['normal_reference'], dtype=float)
    if x.ndim==1:
        x=x[:,None]
    if ref.ndim==1:
        ref=ref[:,None]
    center = np.median(ref, axis=0)
    scale = 1.4826 * np.median(np.abs(ref-center), axis=0)
    scale = np.where(scale>1e-12,scale,np.std(ref,axis=0))
    scale = np.where(scale>1e-12,scale,1.)
    z = (x-center)/scale
    # Reuse the baseline onset evidence floor (3.5 sigma, three samples).
    # Include only sustained shifts; isolated spikes are not competing roots.
    onsets = {}
    for j,name in enumerate(inputs['channel_names']):
        active = np.abs(z[:,j])>=3.5
        indices = np.flatnonzero(np.convolve(active.astype(int),np.ones(3,dtype=int),mode='valid')==3)
        for idx in indices:
            tail = z[int(idx):,j]
            if len(tail)>=6 and abs(float(np.median(tail)))>=3.5:
                onsets[str(name)]=int(idx)
                break
    assessment['reference_based_onsets']=onsets
    if len(onsets)>1:
        first=min(onsets.values())
        competing=[name for name,idx in onsets.items() if idx-first<=3]
        if len(competing)>1:
            assessment.update(reason='simultaneous_process_changes_root_unidentifiable',competing_channels=competing)
    return assessment
