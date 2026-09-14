"use client";

import { useEffect, useState } from "react";

type HealthStatus = {
  online: boolean;
  readyMedicineCount: number | null;
  catalogMedicineCount: number | null;
};

export function SystemStatus() {
  const [health, setHealth] = useState<HealthStatus>({
    online: false,
    readyMedicineCount: null,
    catalogMedicineCount: null,
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
          readyMedicineCount:
            typeof record.ready_medicine_count === "number"
              ? record.ready_medicine_count
              : null,
          catalogMedicineCount:
            typeof record.catalog_medicine_count === "number"
              ? record.catalog_medicine_count
              : null,
        });
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") return;
        setHealth({
          online: false,
          readyMedicineCount: null,
          catalogMedicineCount: null,
        });
      }
    }

    void loadHealth();
    return () => controller.abort();
  }, []);

  return (
    <div
      className="hidden items-center gap-3 rounded-full border border-red-900/10 bg-white/65 px-4 py-2.5 text-[0.61rem] font-extrabold tracking-[0.09em] text-red-950/70 uppercase shadow-status backdrop-blur-xl sm:flex"
      aria-label={`Sistem durumu: ${health.online ? "hazır" : "API bağlantısı bekleniyor"}; ${health.readyMedicineCount ?? 0} yanıtlanabilir, ${health.catalogMedicineCount ?? 0} katalog ilacı`}
      title={`${health.readyMedicineCount?.toLocaleString("tr-TR") ?? "—"} yanıtlanabilir / ${health.catalogMedicineCount?.toLocaleString("tr-TR") ?? "—"} katalog ilacı`}
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
        {health.readyMedicineCount?.toLocaleString("tr-TR") ?? "—"} hazır ilaç
      </span>
    </div>
  );
}
