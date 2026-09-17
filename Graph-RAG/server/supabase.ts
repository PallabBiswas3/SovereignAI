import { createClient } from "@supabase/supabase-js";
import dotenv from "dotenv";
import path from "path";

// Support both local TypeScript execution and compiled server/dist execution.
// Existing environment variables always win; no file is allowed to override them.
const envCandidates = [
  path.resolve(__dirname, ".env"),
  path.resolve(__dirname, "..", ".env"),
  path.resolve(process.cwd(), ".env"),
  path.resolve(process.cwd(), "server", ".env"),
];

for (const envPath of [...new Set(envCandidates)]) {
  dotenv.config({ path: envPath, override: false });
}

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServerKey =
  process.env.SUPABASE_SECRET_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl) {
  throw new Error("SUPABASE_URL is missing from the server environment.");
}

if (!supabaseServerKey) {
  throw new Error(
    "A server-only Supabase credential is required. Set SUPABASE_SECRET_KEY (preferred) or SUPABASE_SERVICE_ROLE_KEY. Do not use SUPABASE_ANON_KEY for backend writes."
  );
}

export const supabase = createClient(supabaseUrl, supabaseServerKey, {
  auth: {
    persistSession: false,
    autoRefreshToken: false,
  },
});
