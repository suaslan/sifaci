import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "Şifacı AI — Kaynağı Görünür İlaç Bilgisi",
  description:
    "Doğrulanmış yerel kaynaklar üzerinden şeffaf ve güvenilir ilaç bilgisi.",
};

export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: "#fafaf9",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="tr">
      <body>{children}</body>
    </html>
  );
}
