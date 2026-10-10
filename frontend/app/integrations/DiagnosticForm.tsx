"use client";

import { useEffect, useState } from "react";

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

function Fields({ schema, values, prefix, change, upload }: {
  schema: Schema; values: Record<string, string>; prefix: string;
  change: (key: string, value: string) => void;
  upload: (key: string, file: File, spec: Schema) => Promise<void>;
}) {
  return <>{Object.entries(schema.properties ?? {}).map(([name, field]) => {
    const spec = resolve(field, schema);
    const key = `${prefix}.${name}`;
    const required = schema.required?.includes(name) ?? false;
    const structured = spec.type === "array" || spec.type === "object";
    return <label key={key}>{name}{required ? " *" : " (optional)"}
      {spec.description && <small>{spec.description}</small>}
      {spec.const !== undefined ? <input value={String(spec.const)} readOnly aria-label={name}/>
        : spec.enum ? <select required={required} value={values[key] ?? ""} onChange={(event) => change(key, event.target.value)}><option value="">Select {name}</option>{spec.enum.map((item) => <option key={item}>{item}</option>)}</select>
        : structured ? <><textarea required={required} value={values[key] ?? ""} onChange={(event) => change(key, event.target.value)} spellCheck={false} placeholder={spec.type === "array" ? (spec.items?.type === "array" ? "[[1, 2], [3, 4], ...]" : "[value, value, ...]") : "JSON object using the fields listed below"}/>
          {spec.type === "array" && <><small>Import JSON array; numeric arrays also accept headerless CSV. Rows = observations, columns = channels. No automatic unit conversion.</small><input aria-label={`Import ${name}`} type="file" accept=".json,.csv" onChange={(event) => { const file = event.target.files?.[0]; if (file) void upload(key, file, spec); event.target.value = ""; }}/></>}
          {spec.properties && <small>Required keys: {(spec.required ?? []).join(", ")}. All values numeric.</small>}
          {spec.minItems && <small>Minimum {spec.minItems} rows/samples. This is a validation floor, not a guarantee of diagnostic adequacy.</small>}</>
        : <input required={required} type={spec.type === "number" || spec.type === "integer" ? "number" : "text"} step="any" value={values[key] ?? ""} onChange={(event) => change(key, event.target.value)} aria-label={name}/>
      }
    </label>;
  })}</>;
}

export default function DiagnosticForm({ catalogue, onChange }: { catalogue: Catalogue; onChange: (value: Diagnostic | null) => void }) {
  const [workflowId, setWorkflowId] = useState(catalogue.workflows[0]?.id ?? "");
  const [values, setValues] = useState<Record<string, string>>({});
  const [fileError, setFileError] = useState("");
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
    try {
      if (file.size > 10 * 1024 * 1024) throw new Error("Input file exceeds the 10 MiB limit");
      const text = await file.text();
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
    } catch (caught) { change(key, ""); setFileError(caught instanceof Error ? caught.message : "File import failed"); }
  }

  return <fieldset className="diagnosticInputs"><legend>Versioned diagnostic data — {catalogue.contract_version}</legend>
    <label>WORKFLOW<select value={workflowId} onChange={(event) => { setWorkflowId(event.target.value); setValues({}); setFileError(""); onChange(null); }}>{catalogue.workflows.map((item) => <option key={item.id} value={item.id}>{item.domain} / {item.task}</option>)}</select></label>
    {workflow && <>
      <p>Policy: <code>{workflow.policy_ref}</code>. Required fields are marked *. Switching workflows clears previous inputs.</p>
      <h3>Asset and collection metadata</h3><Fields schema={catalogue.metadata_schema} values={values} prefix="metadata" change={change} upload={upload}/>
      <h3>Measurements</h3><Fields schema={workflow.input_schema} values={values} prefix="inputs" change={change} upload={upload}/>
      {workflow.model_slots.length > 0 && <h3>Registered model references</h3>}
      {workflow.model_slots.map((slot) => <label key={slot}>{slot}{workflow.required_model_slots.includes(slot) ? " *" : " (optional)"}<input required={workflow.required_model_slots.includes(slot)} value={values[`model.${slot}`] ?? ""} onChange={(event) => change(`model.${slot}`, event.target.value)}/></label>)}
      <p>Model IDs refer to server-registered artifacts, not filenames. Input validation does not establish model availability or factual correctness.</p>
    </>}
    {error && <p role="status">{error}</p>}{fileError && <p className="error" role="alert">{fileError}</p>}
  </fieldset>;
}
