"use client";

import { useEffect, useId, useRef, useState } from "react";

type Schema = {
  type?: string; title?: string; description?: string; const?: unknown;
  enum?: string[]; properties?: Record<string, Schema>; required?: string[];
  $ref?: string; $defs?: Record<string, Schema>; anyOf?: Schema[];
  items?: Schema; minItems?: number; minimum?: number; exclusiveMinimum?: number;
};
type Workflow = {
  id: string; domain: string; task: string; policy_ref: string;
  input_schema: Schema; model_slots: string[]; required_model_slots: string[];
};
export type Catalogue = { contract_version: string; metadata_schema: Schema; workflows: Workflow[] };
export type Diagnostic = {
  contract_version: string; domain: string; task: string; policy_ref: string;
  inputs: Record<string, unknown>; model_refs: Record<string, string>;
  run_context: { metadata: Record<string, unknown> };
};

function resolve(schema: Schema, root: Schema): Schema {
  if (schema.$ref) {
    if (!schema.$ref.startsWith("#/$defs/")) throw new Error("Unsupported contract reference");
    const target = root.$defs?.[schema.$ref.slice(8)];
    if (!target) throw new Error("Missing contract definition");
    return { ...resolve(target, root), description: schema.description ?? target.description };
  }
  if (schema.anyOf) return resolve(schema.anyOf.find((item) => item.type !== "null") ?? schema, root);
  return schema;
}

function decode(schema: Schema, root: Schema, values: Record<string, string>, prefix: string): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [name, field] of Object.entries(schema.properties ?? {})) {
    const spec = resolve(field, root);
    const value = values[`${prefix}.${name}`]?.trim() ?? "";
    if (spec.const !== undefined) { result[name] = spec.const; continue; }
    if (!value) {
      if (schema.required?.includes(name)) throw new Error(`${name} is required`);
      continue;
    }
    if (["array", "object"].includes(spec.type ?? "")) {
      try { result[name] = JSON.parse(value); }
      catch { throw new Error(`${name}: enter valid JSON or import a matching file`); }
    } else if (["number", "integer"].includes(spec.type ?? "")) {
      const number = Number(value);
      if (!Number.isFinite(number)) throw new Error(`${name} must be a finite number`);
      result[name] = number;
    } else result[name] = value;
  }
  return result;
}

function fieldTitle(name: string) {
  return name.replace(/_/g, " ").replace(/\b(utc|id|hz|rpm|rul|ah)\b/g, (word) => word.toUpperCase());
}

function arraySummary(value: string): string {
  if (!value.trim()) return "No data added";
  try {
    const data = JSON.parse(value);
    if (!Array.isArray(data)) return "Expected a JSON array";
    return Array.isArray(data[0]) ? `${data.length} rows · ${data[0].length} columns` : `${data.length} values`;
  } catch { return "Invalid JSON"; }
}

function Fields({ schema, values, prefix, change, upload }: {
  schema: Schema; values: Record<string, string>; prefix: string;
  change: (key: string, value: string) => void;
  upload: (key: string, file: File, spec: Schema) => Promise<void>;
}) {
  const fieldId = useId();
  return <>{Object.entries(schema.properties ?? {}).map(([name, field]) => {
    const spec = resolve(field, schema);
    const key = `${prefix}.${name}`;
    const required = schema.required?.includes(name) ?? false;
    const structured = spec.type === "array" || spec.type === "object";
    const id = `${fieldId}-${name}`;
    const numericArray = spec.type === "array" && ["number", "integer"].includes((spec.items?.type === "array" ? spec.items.items : spec.items)?.type ?? "");
    return <div key={key} className={`diagnosticField ${structured ? "diagnosticFieldWide" : ""}`}>
      <label htmlFor={id}><span>{fieldTitle(name)}</span><span className="fieldRequirement">{spec.const !== undefined ? "Fixed" : required ? "Required" : "Optional"}</span></label>
      <small className="fieldKey">{name}</small>
      {spec.const !== undefined ? <input id={id} value={String(spec.const)} readOnly aria-label={name} aria-describedby={`${id}-help`}/>
        : spec.enum ? <select id={id} aria-label={name} aria-describedby={`${id}-help`} required={required} value={values[key] ?? ""} onChange={(event) => change(key, event.target.value)}><option value="">Select {fieldTitle(name)}</option>{spec.enum.map((item) => <option key={item}>{item}</option>)}</select>
        : structured ? <><textarea id={id} aria-label={name} aria-describedby={`${id}-help`} required={required} value={values[key] ?? ""} onChange={(event) => change(key, event.target.value)} spellCheck={false} placeholder={spec.type === "array" ? (spec.items?.type === "array" ? "[[1, 2], [3, 4], ...]" : "[value, value, ...]") : "JSON object using the fields listed below"}/>
          {spec.type === "array" && <div className="dataImport"><input aria-label={`Import ${name}`} type="file" accept={numericArray ? ".json,.csv" : ".json"} onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(key, file, spec); event.target.value = ""; }}/><span>{arraySummary(values[key] ?? "")}</span></div>}
        </>
        : <input id={id} required={required} type={spec.type === "number" || spec.type === "integer" ? "number" : "text"} step={spec.type === "integer" ? "1" : "any"} value={values[key] ?? ""} onChange={(event) => change(key, event.target.value)} aria-label={name} aria-describedby={`${id}-help`}/>
      }
      <div id={`${id}-help`} className="fieldHelp">
        {spec.description && <small>{spec.description}</small>}
        {spec.type === "array" && <small>{numericArray ? "JSON or headerless numeric CSV. " : "JSON array. "}{spec.items?.type === "array" ? "Rows = observations; columns = channels. " : "One value per entry. "}No automatic unit conversion.</small>}
        {spec.properties && <small>Required keys: {(spec.required ?? []).join(", ")}. All values numeric.</small>}
        {spec.minItems && <small>Minimum {spec.minItems} rows/samples; not a guarantee of diagnostic adequacy.</small>}
      </div>
    </div>;
  })}</>;
}

export default function DiagnosticForm({ catalogue, onChange }: { catalogue: Catalogue; onChange: (value: Diagnostic | null) => void }) {
  const [workflowId, setWorkflowId] = useState(catalogue.workflows[0]?.id ?? "");
  const [values, setValues] = useState<Record<string, string>>({});
  const [fileError, setFileError] = useState("");
  const importGeneration = useRef(0);
  const workflow = catalogue.workflows.find((item) => item.id === workflowId);
  let diagnostic: Diagnostic | null = null;
  let error = "";
  try {
    if (workflow) {
      const refs: Record<string, string> = {};
      for (const slot of workflow.model_slots) {
        const ref = values[`model.${slot}`]?.trim();
        if (ref) refs[slot] = ref;
        else if (workflow.required_model_slots.includes(slot)) throw new Error(`${slot}: registered model reference required`);
      }
      diagnostic = {
        contract_version: catalogue.contract_version, domain: workflow.domain, task: workflow.task,
        policy_ref: workflow.policy_ref, inputs: decode(workflow.input_schema, workflow.input_schema, values, "inputs"),
        model_refs: refs, run_context: { metadata: decode(catalogue.metadata_schema, catalogue.metadata_schema, values, "metadata") },
      };
    }
  } catch (caught) { error = caught instanceof Error ? caught.message : "Complete required fields"; }
  // Emit only structurally complete JSON. The server checks all semantic constraints.
  const serialized = JSON.stringify(diagnostic);
  useEffect(() => { onChange(JSON.parse(serialized)); }, [serialized, onChange]);

  function change(key: string, value: string) { setValues((previous) => ({ ...previous, [key]: value })); }
  async function upload(key: string, file: File, spec: Schema) {
    const generation = importGeneration.current;
    try {
      if (file.size > 10 * 1024 * 1024) throw new Error("Input file exceeds the 10 MiB limit");
      const text = await file.text();
      if (generation !== importGeneration.current) return;
      let value: unknown;
      if (file.name.toLowerCase().endsWith(".csv")) {
        const matrix = spec.items?.type === "array";
        const item = matrix ? spec.items?.items : spec.items;
        if (item?.type !== "number" && item?.type !== "integer") throw new Error("CSV is supported only for numeric fields; use a JSON array for names/timestamps");
        const rows = text.trim().split(/\r?\n/).map((line) => line.split(",").map((cell) => {
          if (!cell.trim() || !Number.isFinite(Number(cell))) throw new Error("CSV must contain finite numbers only, no headers or blank cells");
          return Number(cell);
        }));
        if (!matrix && rows.some((row) => row.length !== 1)) throw new Error("Vector CSV must have exactly one column");
        value = matrix ? rows : rows.map((row) => row[0]);
      } else value = JSON.parse(text);
      if (!Array.isArray(value)) throw new Error("File must contain an array, not a full diagnostic request");
      change(key, JSON.stringify(value)); setFileError("");
    } catch (caught) { if (generation !== importGeneration.current) return; change(key, ""); setFileError(caught instanceof Error ? caught.message : "File import failed"); }
  }

  // Completion is input presence only; the backend still validates semantics.
  const requiredFields = workflow ? [
    ...(catalogue.metadata_schema.required ?? []).filter((key) => resolve(catalogue.metadata_schema.properties![key], catalogue.metadata_schema).const === undefined).map((key) => `metadata.${key}`),
    ...(workflow.input_schema.required ?? []).filter((key) => resolve(workflow.input_schema.properties![key], workflow.input_schema).const === undefined).map((key) => `inputs.${key}`),
    ...workflow.required_model_slots.map((key) => `model.${key}`),
  ] : [];
  const completed = requiredFields.filter((key) => values[key]?.trim()).length;

  return <fieldset className="diagnosticInputs"><legend>Diagnostic input</legend>
    <div className="contractHeading"><span className="sectionEyebrow">01 / SELECT WORKFLOW</span><code>{catalogue.contract_version}</code></div>
    <label className="workflowSelect">Diagnostic workflow<select value={workflowId} onChange={(event) => { importGeneration.current += 1; setWorkflowId(event.target.value); setValues({}); setFileError(""); onChange(null); }}>{catalogue.workflows.map((item) => <option key={item.id} value={item.id}>{item.domain} / {item.task}</option>)}</select></label>
    {workflow && <>
      <p className="contractNote">Policy: <code>{workflow.policy_ref}</code>. Switching workflows clears previous inputs.</p>
      <div className="inputProgress"><span>{completed} of {requiredFields.length} required fields entered</span><progress aria-label="Required fields entered" value={completed} max={requiredFields.length || 1}/></div>
      <section className="diagnosticSection"><div className="diagnosticSectionHeading"><span>02</span><div><h3>Asset &amp; collection</h3><p>Identify the equipment and when the data was recorded.</p></div></div><div className="diagnosticFieldGrid"><Fields schema={catalogue.metadata_schema} values={values} prefix="metadata" change={change} upload={upload}/></div></section>
      <section className="diagnosticSection"><div className="diagnosticSectionHeading"><span>03</span><div><h3>Measurements</h3><p>Paste data or import individual arrays. Keep the original units.</p></div></div><div className="diagnosticFieldGrid"><Fields schema={workflow.input_schema} values={values} prefix="inputs" change={change} upload={upload}/></div></section>
      {workflow.model_slots.length > 0 && <section className="diagnosticSection"><div className="diagnosticSectionHeading"><span>04</span><div><h3>Registered models</h3><p>Use server-registered IDs, not filenames.</p></div></div><div className="diagnosticFieldGrid">
        {workflow.model_slots.map((slot) => <label className="modelReference" key={slot}>{fieldTitle(slot)}<small>{slot} · {workflow.required_model_slots.includes(slot) ? "Required" : "Optional"}</small><input aria-label={slot} required={workflow.required_model_slots.includes(slot)} value={values[`model.${slot}`] ?? ""} onChange={(event) => change(`model.${slot}`, event.target.value)}/></label>)}
      </div></section>}
      <p className="contractNote">Input completion does not establish model availability or factual correctness. Server validation runs before analysis.</p>
    </>}
    <p className={`diagnosticReadiness ${diagnostic ? "complete" : ""}`} role="status">{error ? `Next: ${error}` : "Inputs complete — ready for server validation."}</p>{fileError && <p className="error" role="alert">{fileError}</p>}
  </fieldset>;
}
