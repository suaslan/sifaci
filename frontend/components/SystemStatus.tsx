"use client";

import { useEffect, useState } from "react";

type HealthStatus = {
  online: boolean;
  medicineCount: number | null;
};

export function SystemStatus() {
  const [health, setHealth] = useState<HealthStatus>({
    online: false,
    medicineCount: null,
  });

  useEffect(() => {
    const controller = new AbortController();

    async function loadHealth() {
      try {
        const response = await fetch("/api/health", {
          cache: "no-store",
          signal: controller.signal,
        });
        const payload: unknown = await response.json();
        if (typeof payload !== "object" || payload === null) return;
        const record = payload as Record<string, unknown>;
        setHealth({
          online: response.ok && record.online === true,
          medicineCount:
            typeof record.medicine_count === "number" ? record.medicine_count : null,
        });
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") return;
        setHealth({ online: false, medicineCount: null });
      }
    }

    void loadHealth();
    return () => controller.abort();
  }, []);

  return (
    <div
      className="hidden items-center gap-3 rounded-full border border-red-900/10 bg-white/65 px-4 py-2.5 text-[0.61rem] font-extrabold tracking-[0.09em] text-red-950/70 uppercase shadow-status backdrop-blur-xl sm:flex"
      aria-label={`Sistem durumu: ${health.online ? "hazır" : "API bağlantısı bekleniyor"}`}
    >
      <span className="flex items-center gap-2">
        <span
          className={`status-dot ${health.online ? "" : "status-dot--waiting"}`}
          aria-hidden="true"
        />
        Yerel AI
      </span>
      <span className="hidden h-3 w-px bg-navy/10 sm:block" aria-hidden="true" />
      <span className="hidden sm:inline">Cihazda</span>
      <span className="hidden h-3 w-px bg-navy/10 md:block" aria-hidden="true" />
      <span className="hidden md:inline">RAG aktif</span>
      <span className="hidden h-3 w-px bg-navy/10 lg:block" aria-hidden="true" />
      <span className="hidden lg:inline">
        {health.medicineCount?.toLocaleString("tr-TR") ?? "—"} ilaç
      </span>
    </div>
  );
}
