import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  // Allow either loopback hostname during local development. Browser-facing
  // configuration still uses localhost consistently so Strict auth cookies
  // are sent to the API.
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
