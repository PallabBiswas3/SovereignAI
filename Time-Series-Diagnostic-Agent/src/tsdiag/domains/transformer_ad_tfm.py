from __future__ import annotations

from typing import Any

from .runners import TransformerDiagnosticPipeline


class ADTFMTransformerDiagnosticPipeline:
    """Use a raw-waveform AD-TFM-AT classifier inside the existing transformer pipeline.

    The existing transformer path still computes synchronization, denoising,
    multisensor/cyclostationary evidence and optional deterministic electrical
    rules. This adapter replaces only the learned classifier with the paper's
    raw Ua/Ub/Uc/Ia/Ib/Ic time-frequency recurrent model, so the project keeps
    its evidence/verification and abstention behavior rather than becoming a
    standalone opaque neural classifier.
    """

    def __init__(self, classifier):
        if not callable(classifier):
            raise TypeError("classifier must be a trained callable AD-TFM classifier")
        self.classifier = classifier
        self.base_pipeline = TransformerDiagnosticPipeline()

    def run(self, signal_matrix, sampling_rate_hz, sensor_positions, **context: Any):
        deep_result = self.classifier(signal_matrix)

        # TransformerDiagnosticPipeline's established model hook receives the
        # spectral representation. The AD-TFM model has already consumed the
        # synchronized raw six-channel waveform, so this tiny adapter forwards
        # its frozen result through the existing evidence/decision machinery.
        def trained_model_adapter(_feature_image):
            return deep_result

        return self.base_pipeline.run(
            signal_matrix,
            sampling_rate_hz,
            sensor_positions,
            **context,
            trained_image_model=trained_model_adapter,
        )
