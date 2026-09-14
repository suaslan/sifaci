import type { MedicineAnswer } from "@/types/medicine";

type ErrorPayload = {
  error?: string;
};

export class MedicineApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MedicineApiError";
  }
}

const MAX_DISPLAY_ANSWER_CHARS = 6_000;
const missingInformationMessage =
  "Bu soruyu yanıtlamak için bilgi tabanında yeterli kaynak bulunamadı.";

function sanitizeAnswerText(value: string): string {
  const withoutSvg = value
    .replace(/<svg\b[^>]*>[\s\S]*?<\/svg\s*>/gi, "")
    .replace(/&lt;svg\b[\s\S]*?&lt;\/svg\s*&gt;/gi, "")
    .replace(/<\/?(?:svg|path|circle|rect|line|polyline|polygon|ellipse|g)\b[^>]*>/gi, "")
    .replace(/&lt;\/?(?:svg|path|circle|rect|line|polyline|polygon|ellipse|g)\b.*?&gt;/gi, "")
    .replace(/^\s*svg\s*$/gim, "")
    .trim();

  if (!withoutSvg || withoutSvg.length > MAX_DISPLAY_ANSWER_CHARS) {
    return missingInformationMessage;
  }
  return withoutSvg;
}

export async function askMedicineAssistant(
  question: string,
  signal?: AbortSignal,
): Promise<MedicineAnswer> {
  const response = await fetch("/api/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });

  const payload: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    const errorPayload = payload as ErrorPayload;
    throw new MedicineApiError(
      errorPayload.error || "Şifacı AI şu anda yanıt oluşturamadı.",
    );
  }

  const answer = payload as Partial<MedicineAnswer>;
  if (
    typeof answer.answer !== "string" ||
    !Array.isArray(answer.sources) ||
    !answer.sources.every((source) => typeof source === "string") ||
    typeof answer.disclaimer !== "string"
  ) {
    throw new MedicineApiError("API geçersiz bir cevap döndürdü.");
  }

  return {
    answer: sanitizeAnswerText(answer.answer),
    medicine: typeof answer.medicine === "string" ? answer.medicine : null,
    sources: answer.sources,
    disclaimer: answer.disclaimer,
    suggestions: Array.isArray(answer.suggestions)
      ? [...new Set(answer.suggestions.filter((item): item is string => typeof item === "string"))]
          .map((item) => item.trim())
          .filter(Boolean)
          .slice(0, 5)
      : [],
  };
}
