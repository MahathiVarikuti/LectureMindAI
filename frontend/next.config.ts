import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Required by the production Docker image (frontend/Dockerfile copies .next/standalone).
  output: "standalone",
};

export default nextConfig;
