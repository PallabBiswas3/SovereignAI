import type { DiagnosticUpdate } from "../../lib/diagnostic-update";

export default function DiagnosticUpdates({ updates }: { updates?: DiagnosticUpdate[] }) {
  if (!updates?.length) return null;
  return <div className="diagnosticUpdates" aria-label="Additional diagnostic analysis">
    {updates.map((update, index) => <article key={`${update.model_ref}-${index}`}>
      <h3>Additional diagnostic analysis</h3>
      <b>{update.status === "abstained" ? "Additional analysis withheld" : "Analysis completed"}</b>
      {update.synthetic_artifact && <p>Synthetic demonstration model</p>}
      <p>{update.summary}</p>
      <p>{update.calibrated ? "Method has fitted calibration; this is not an individual correctness probability." : "Uncertainty is model-based or not calibrated."}</p>
      {!!update.prediction_set?.length && <p>Prediction set: {update.prediction_set.join(", ")}</p>}
      {update.point_estimate != null && <div className="metric"><span>Remaining useful life</span><b>{update.point_estimate.toFixed(2)} {update.unit}</b></div>}
      {update.interval && <div className="metric"><span>Forecast interval</span><b>{update.interval[0].toFixed(2)}–{update.interval[1].toFixed(2)} {update.unit}</b></div>}
      {update.details?.alarm != null && <p>Additional alarm: {update.details.alarm === true ? "yes" : "no"}</p>}
      {update.details?.policy_hold_remains === true && <p>The diagnostic policy still requires review.</p>}
      <details><summary>Analysis provenance</summary><p>Model: {update.model_ref} · Version: {update.version}</p><p>Method: {update.method}</p>{update.calibration_dataset && <p>Calibration: {update.calibration_dataset}</p>}</details>
    </article>)}
  </div>;
}
