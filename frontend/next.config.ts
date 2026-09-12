import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  allowedDevOrigins: ["localhost", "127.0.0.1"],
  // Next.js distDir proje disinda olamaz. setup_d_drive.ps1 bu yolu
  // D:\SifaciAI\cache\nextjs hedefine baglanan bir Windows junction yapar.
  turbopack: {
    root: process.cwd(),
  },
};

export default nextConfig;
