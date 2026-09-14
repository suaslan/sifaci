import { NextResponse } from "next/server";

import type { MedicineAnswer } from "@/types/medicine";


export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const backendUrl = (process.env.SIFACI_BACKEND_URL || "http://127.0.0.1:8000").replace(
  /\/$/,
  "",
);
const configuredTimeout = Number(process.env.SIFACI_REQUEST_TIMEOUT_MS || 300_000);
const backendTimeoutMs =
  Number.isFinite(configuredTimeout) && configuredTimeout >= 10_000
    ? configuredTimeout
    : 300_000;

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Geçerli bir soru gönderin." }, { status: 400 });
  }

  const question =
    typeof body === "object" &&
    body !== null &&
    "question" in body &&
    typeof body.question === "string"
      ? body.question.trim()
      : "";

  if (!question) {
    return NextResponse.json({ error: "Soru boş olamaz." }, { status: 400 });
  }
  if (question.length > 1_000) {
    return NextResponse.json(
      { error: "Soru en fazla 1000 karakter olabilir." },
      { status: 400 },
    );
  }

  try {
    const backendResponse = await fetch(`${backendUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
      cache: "no-store",
      signal: AbortSignal.timeout(backendTimeoutMs),
    });
    const payload: unknown = await backendResponse.json().catch(() => ({}));

    if (!backendResponse.ok) {
      const detail =
        typeof payload === "object" && payload !== null && "detail" in payload
          ? String(payload.detail)
          : "Python RAG servisi yanıt oluşturamadı.";
      return NextResponse.json({ error: detail }, { status: backendResponse.status });
    }

    return NextResponse.json(payload as MedicineAnswer);
  } catch (error) {
    const message =
      error instanceof Error && error.name === "TimeoutError"
        ? "Yerel model yanıtı zaman aşımına uğradı."
        : "Python API çalışmıyor. Önce 'python api.py' komutunu başlatın.";
    return NextResponse.json({ error: message }, { status: 503 });
  }
}
