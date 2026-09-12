import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const backendUrl = (process.env.SIFACI_BACKEND_URL || "http://127.0.0.1:8000").replace(
  /\/$/,
  "",
);

export async function GET() {
  try {
    const response = await fetch(`${backendUrl}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
    const payload: unknown = await response.json().catch(() => ({}));
    if (!response.ok) {
      return NextResponse.json({ online: false, medicine_count: 0 }, { status: 503 });
    }
    return NextResponse.json({ online: true, ...(payload as object) });
  } catch {
    return NextResponse.json({ online: false, medicine_count: 0 }, { status: 503 });
  }
}
