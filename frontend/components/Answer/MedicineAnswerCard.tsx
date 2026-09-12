"use client";

import {
  AlertTriangle,
  BookOpenText,
  CheckCircle2,
  Clock3,
  LoaderCircle,
  Pill,
  Search,
  SearchX,
  ShieldCheck,
} from "lucide-react";
import { motion, useReducedMotion } from "motion/react";

import type { MedicineAnswer } from "@/types/medicine";

type MedicineAnswerCardProps = {
  question: string;
  loading: boolean;
  result: MedicineAnswer | null;
  error: string | null;
  responseTimeMs?: number | null;
  onSuggestionSelect?: (suggestion: string) => void;
};

function normalizeAnswerLine(line: string) {
  return line
    .replace(/^#{1,4}\s*/, "")
    .replace(/^[-*•]\s*/, "")
    .replace(/^\*\*(.*?)\*\*:?$/, "$1")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .trim();
}

function AnswerBody({ answer }: { answer: string }) {
  const lines = answer
    .split("\n")
    .map(normalizeAnswerLine)
    .filter(Boolean);

  return (
    <div className="space-y-3">
      {lines.map((line, index) => {
        const isWarning = /(önemli güvenlik|uyarı|dikkat|ciddi yan etki|acil)/i.test(line);
        return (
          <div
            key={`${index}-${line.slice(0, 28)}`}
            className={
              isWarning
                ? "flex gap-3 rounded-2xl border border-amber-300/60 bg-amber-50/75 p-4 text-stone-800"
                : "flex gap-3 text-stone-700"
            }
          >
            {isWarning ? (
              <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-700" aria-hidden="true" />
            ) : (
              <span className="mt-[0.65rem] h-1.5 w-1.5 shrink-0 rounded-full bg-red-700" aria-hidden="true" />
            )}
            <p className="text-[0.95rem] leading-7 font-medium">{line}</p>
          </div>
        );
      })}
    </div>
  );
}

export function MedicineAnswerCard({
  question,
  loading,
  result,
  error,
  responseTimeMs,
  onSuggestionSelect,
}: MedicineAnswerCardProps) {
  const reducedMotion = Boolean(useReducedMotion());
  const suggestions = result?.suggestions ?? [];
  const hasSources = Boolean(result?.sources.length);
  const isEmptyResult = Boolean(
    result &&
      result.sources.length === 0 &&
      /(bulunamadı|henüz .*işlenmemiş)/i.test(result.answer),
  );
  const responseSeconds = responseTimeMs == null ? null : (responseTimeMs / 1_000).toFixed(1);

  if (!loading && !result && !error) return null;

  return (
    <motion.section
      initial={reducedMotion ? false : { opacity: 0, y: 34, scale: 0.985 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: reducedMotion ? 0 : 0.58, ease: [0.22, 1, 0.36, 1] }}
      className="answer-crystal mt-8 overflow-hidden rounded-[1.6rem] border border-red-950/10 border-l-[5px] border-l-red-800 bg-white/78 text-left shadow-answer backdrop-blur-2xl"
      aria-live="polite"
      aria-busy={loading}
    >
      <div className="flex flex-col gap-4 border-b border-red-950/10 bg-white/50 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-7">
        <div className="flex min-w-0 items-center gap-3">
          <span className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-red-800 text-white shadow-[0_8px_20px_rgba(153,27,27,0.22)]">
            {loading ? (
              <LoaderCircle className="h-5 w-5 animate-spin" aria-hidden="true" />
            ) : isEmptyResult ? (
              <SearchX className="h-5 w-5" aria-hidden="true" />
            ) : (
              <Pill className="h-5 w-5" aria-hidden="true" />
            )}
          </span>
          <div className="min-w-0">
            <p className="text-[0.65rem] font-black tracking-[0.17em] text-red-800 uppercase">
              Şifacı AI / Kaynaklı yanıt
            </p>
            {question && (
              <h2 className="mt-1 truncate text-base font-black tracking-[-0.025em] text-stone-900 sm:text-lg">
                {question.toLocaleUpperCase("tr-TR")}
              </h2>
            )}
          </div>
        </div>

        {!loading && result && (
          <span
            className={`inline-flex w-fit items-center gap-2 rounded-full px-3 py-2 text-xs font-extrabold ${
              hasSources ? "bg-emerald-50 text-emerald-800" : "bg-stone-100 text-stone-600"
            }`}
          >
            {hasSources ? <CheckCircle2 className="h-4 w-4" aria-hidden="true" /> : <SearchX className="h-4 w-4" aria-hidden="true" />}
            {hasSources ? "Güvenilirlik: Kaynak doğrulandı" : "Belge eşleşmesi yok"}
          </span>
        )}
      </div>

      {loading && (
        <div className="space-y-4 px-6 py-8" role="status">
          <p className="text-sm font-semibold text-stone-700">Kayıtlı kaynaklar taranıyor…</p>
          <span className="block h-2.5 w-full animate-pulse rounded-full bg-red-100" />
          <span className="block h-2.5 w-5/6 animate-pulse rounded-full bg-red-100/75" />
          <span className="block h-2.5 w-2/3 animate-pulse rounded-full bg-red-100/50" />
        </div>
      )}

      {error && !loading && (
        <div className="flex gap-3 px-6 py-8 text-sm leading-6 text-stone-700" role="alert">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-700" aria-hidden="true" />
          <div>
            <p className="font-black text-stone-900">Yanıt alınamadı</p>
            <p className="mt-1">{error}</p>
          </div>
        </div>
      )}

      {result && !loading && (
        <>
          <div className="px-5 py-7 sm:px-7">
            {isEmptyResult ? (
              <div className="flex gap-3 rounded-2xl bg-stone-50 p-4 text-stone-700">
                <SearchX className="mt-0.5 h-5 w-5 shrink-0 text-red-800" aria-hidden="true" />
                <p className="text-[0.95rem] leading-7 font-semibold">{result.answer}</p>
              </div>
            ) : (
              <AnswerBody answer={result.answer} />
            )}
          </div>

          {suggestions.length > 0 && (
            <div className="border-t border-red-950/10 px-5 py-5 sm:px-7">
              <div className="flex items-center gap-2 text-xs font-black text-stone-900">
                <Search className="h-4 w-4 text-red-700" aria-hidden="true" strokeWidth={2} />
                En yakın alternatif eşleşmeler
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {suggestions.slice(0, 5).map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => onSuggestionSelect?.(suggestion)}
                    disabled={!onSuggestionSelect}
                    className="rounded-full border border-red-800/20 bg-red-50/65 px-3.5 py-2 text-xs font-semibold text-red-900 transition-colors hover:border-red-700/50 hover:bg-red-100/75 disabled:cursor-default"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="border-t border-red-950/10 bg-stone-50/70 px-5 py-5 sm:px-7">
            <div className="flex items-center gap-2 text-[0.68rem] font-black tracking-[0.14em] text-red-900 uppercase">
              <BookOpenText className="h-4 w-4 text-red-700" aria-hidden="true" strokeWidth={1.9} />
              Kullanılan kaynaklar
            </div>
            {result.sources.length > 0 ? (
              <div className="mt-3 flex flex-wrap gap-2">
                {result.sources.map((source) => (
                  <span
                    key={source}
                    className="rounded-full border border-red-950/10 bg-white px-3 py-1.5 text-xs font-semibold text-stone-700"
                  >
                    {source}
                  </span>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-xs font-medium text-stone-500">Güvenilir kaynak eşleşmesi bulunamadı.</p>
            )}

            <div className="mt-4 flex flex-col gap-3 border-t border-red-950/10 pt-4 text-xs leading-5 text-stone-500 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex max-w-2xl gap-2.5">
                <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-red-700" aria-hidden="true" />
                <p>{result.disclaimer}</p>
              </div>
              {responseSeconds && (
                <span className="flex shrink-0 items-center gap-1.5 font-bold text-stone-600">
                  <Clock3 className="h-4 w-4 text-red-700" aria-hidden="true" />
                  {responseSeconds}s
                </span>
              )}
            </div>
          </div>
        </>
      )}
    </motion.section>
  );
}
