import { createHash, timingSafeEqual } from "node:crypto";
import type { Request, Response, NextFunction } from "express";
import type { RetrievalAuthorizationScope } from "./retrieval/hybridRetriever";

function matches(value: string, configured: string | undefined): boolean {
  if (!configured || configured.length < 32) return false;
  return timingSafeEqual(createHash("sha256").update(value).digest(),
    createHash("sha256").update(configured).digest());
}

/** The authenticated backend supplies scopes derived from its session principal.
 * A separate operator credential protects legacy global administrative routes.
 * Neither credential belongs in the GraphRAG browser client.
 */
export function serviceAuthentication(req: Request, res: Response, next: NextFunction) {
  const token = req.get("Authorization")?.match(/^Bearer (\S+)$/)?.[1] || "";
  const service = process.env.GRAPHRAG_SERVICE_TOKEN;
  const admin = process.env.GRAPHRAG_ADMIN_TOKEN;
  if (req.path === "/integration/retrieve") {
    if (!service || service.length < 32) return res.status(503).json({ message: "SERVICE_AUTH_UNCONFIGURED" });
    if (!matches(token, service)) return res.status(401).json({ message: "SERVICE_AUTH_REQUIRED" });
  } else {
    // Global graph views, ingestion, evaluation and deletion are operator-only.
    if (!admin || admin === service) return res.status(403).json({ message: "ADMIN_API_DISABLED" });
    if (!matches(token, admin)) return res.status(401).json({ message: "ADMIN_AUTH_REQUIRED" });
  }
  return next();
}

export function validatedScope(value: unknown): RetrievalAuthorizationScope | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const scope = value as Record<string, unknown>;
  const validId = (id: unknown): id is string => typeof id === "string"
    && id.length > 0 && id.length <= 200 && id.trim() === id && !/[\x00-\x1f*]/.test(id);
  const list = (items: unknown): items is string[] => Array.isArray(items) && items.length <= 100
    && items.every(validId) && new Set(items).size === items.length;
  const roles = new Set(["USER", "ENGINEER", "APPROVER", "MANAGER", "AUDITOR", "ADMIN"]);
  if (!validId(scope.organization_id) || !validId(scope.user_id)
    || !list(scope.workspace_ids) || !scope.workspace_ids.length
    || !list(scope.department_ids) || !list(scope.roles) || !scope.roles.length
    || !scope.roles.every(role => roles.has(role))
    || !Number.isInteger(scope.clearance) || Number(scope.clearance) < 0 || Number(scope.clearance) > 3
    || typeof scope.cross_department !== "boolean") return null;
  const values = [scope.organization_id, scope.user_id, String(scope.clearance), scope.cross_department ? "1" : "0",
    ...[...scope.department_ids].sort(), ...[...scope.workspace_ids].sort(), ...[...scope.roles].sort()];
  const fingerprint = createHash("sha256").update(values.join("\x1f")).digest("hex");
  if (scope.fingerprint !== fingerprint) return null;
  // Copy a bounded immutable-in-practice scope; no unexpected client fields.
  return { organization_id: scope.organization_id, user_id: scope.user_id,
    department_ids: [...scope.department_ids], workspace_ids: [...scope.workspace_ids], roles: [...scope.roles],
    clearance: Number(scope.clearance), cross_department: scope.cross_department, fingerprint };
}
