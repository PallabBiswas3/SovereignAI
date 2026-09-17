from ..contracts import DataKind, DomainPack, TaskKind, ToolContract


PROCESS_PACK = DomainPack(
    key="process",
    title="Chemical / Process Plant Diagnostics",
    data_kind=DataKind.MULTIVARIATE_SERIES,
    tasks=(TaskKind.CONDITION_MONITORING, TaskKind.ROOT_CAUSE, TaskKind.FAULT_DIAGNOSIS),
    required_metadata=("channel_names", "sampling_rate_hz"),
    optional_metadata=("normal_reference", "process_topology", "operating_mode", "trained_fault_classifier", "monitoring_method"),
    tools=(
        ToolContract("process_data_quality", "Check missingness, constant sensors, scaling and synchronization.", ("signal_matrix", "channel_names"), outputs=("quality_flags", "channel_statistics"), evidence_fields=("quality_flags",), implementation="tsdiag.tools.data_quality_check"),
        ToolContract("standardize_against_normal", "Normalize current data using healthy-reference statistics.", ("signal_matrix", "normal_reference"), outputs=("standardized_signal", "reference_mean", "reference_scale"), preconditions=("reference and current channels must align",), implementation="tsdiag.tools.standardize_against_normal"),
        ToolContract("pca_monitoring", "Model normal process subspace and compute T2/SPE-style deviations.", ("standardized_signal", "normal_reference"), outputs=("scores", "t2", "spe", "control_limits", "alarm_mask"), evidence_fields=("t2", "spe", "control_limits"), implementation="tsdiag.tools.pca_monitoring"),
        ToolContract("contribution_analysis", "Rank variables contributing most to the current process alarm.", ("signal_matrix", "pca_state", "alarm_mask"), outputs=("variable_contributions", "suspect_variables"), evidence_fields=("suspect_variables", "variable_contributions"), implementation="tsdiag.tools.contribution_analysis"),
        ToolContract("stationarity_analysis", "Determine whether causal analysis assumptions are plausible in the active window.", ("signal_matrix",), outputs=("stationarity_flags", "differencing_recommendations"), evidence_fields=("stationarity_flags",), implementation="tsdiag.tools.stationarity_analysis"),
        ToolContract("granger_causality", "Screen directed predictive relationships among synchronized variables.", ("signal_matrix", "channel_names"), optional_inputs=("maxlag", "alpha"), outputs=("directed_edges", "p_values"), evidence_fields=("directed_edges",), failure_modes=("insufficient_samples", "nonstationary_input", "singular_model"), implementation="tsdiag.tools.granger_causality"),
        ToolContract("causal_graph_filter", "Constrain predictive links using process topology/physics when available.", ("directed_edges",), optional_inputs=("process_topology",), outputs=("filtered_causal_graph", "removed_edges"), evidence_fields=("filtered_causal_graph",), implementation="tsdiag.tools.causal_graph_filter"),
        ToolContract("fault_onset_timing", "Estimate when each affected variable first became abnormal.", ("signal_matrix", "alarm_mask", "channel_names"), outputs=("onset_times", "onset_order"), evidence_fields=("onset_order",), implementation="tsdiag.tools.fault_onset_timing"),
        ToolContract("root_cause_rank", "Combine contribution, causality and onset order to rank root-cause variables.", ("suspect_variables", "filtered_causal_graph", "onset_order"), outputs=("root_cause_ranking", "propagation_paths", "confidence"), evidence_fields=("root_cause_ranking", "propagation_paths"), implementation="tsdiag.tools.root_cause_rank"),
        ToolContract("process_diagnosis", "Convert root-cause evidence into diagnosis or abstention.", ("root_cause_ranking", "propagation_paths"), optional_inputs=("fault_catalog",), outputs=("fault_label", "root_cause", "confidence", "abstain_reason"), evidence_fields=("root_cause", "propagation_paths"), implementation="tsdiag.tools.process_diagnosis"),
    ),
    outputs=("fault_detected", "fault_label", "root_cause", "affected_variables", "propagation_path", "confidence", "evidence"),
    benchmark_targets=("detection_f1", "root_cause_accuracy", "fault_class_accuracy", "false_alarm_rate", "detection_delay"),
)
