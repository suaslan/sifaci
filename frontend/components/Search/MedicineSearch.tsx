"use client";

import { LoaderCircle, Search, Sparkles } from "lucide-react";
import { useId, useRef, useState } from "react";
import type { FormEvent } from "react";

import { SuggestedQueries } from "@/components/Search/SuggestedQueries";

type MedicineSearchProps = {
  onSearch: (question: string) => void | Promise<void>;
  isLoading?: boolean;
};

export function MedicineSearch({ onSearch, isLoading = false }: MedicineSearchProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [question, setQuestion] = useState("");
  const [announcement, setAnnouncement] = useState("");

  const selectSuggestion = (suggestion: string) => {
    setQuestion(suggestion);
    inputRef.current?.focus();
  };

  const submitQuestion = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalizedQuestion = question.trim();
    if (!normalizedQuestion) {
      inputRef.current?.focus();
      return;
    }

    setAnnouncement(`Sorgu gönderildi: ${normalizedQuestion}`);
    void onSearch(normalizedQuestion);
  };

  return (
    <div className="w-full max-w-full space-y-4">
      <form
        onSubmit={submitQuestion}
        className="search-crystal group flex min-h-[5rem] w-full min-w-0 items-center gap-3 rounded-[1.5rem] border-2 border-red-800 bg-white/70 p-2.5 pl-5 shadow-search backdrop-blur-2xl transition-[border-color,box-shadow,transform] duration-300 focus-within:-translate-y-0.5 focus-within:border-red-600 focus-within:shadow-search-focus sm:pl-6"
      >
        <label htmlFor={inputId} className="sr-only">
          İlaca dair öğrenmek istediğiniz soru
        </label>
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-red-50 text-red-800">
          <Search className="h-4.5 w-4.5" strokeWidth={2} aria-hidden="true" />
        </span>
        <input
          id={inputId}
          ref={inputRef}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="İlaca dair ne öğrenmek istiyorsun? Örn: Parol yan etkileri"
          autoComplete="off"
          maxLength={1_000}
          disabled={isLoading}
          className="min-w-0 flex-1 bg-transparent text-[0.95rem] font-semibold text-stone-900 outline-none placeholder:font-medium placeholder:text-stone-500 sm:text-base"
        />
        <button
          type="submit"
          disabled={isLoading}
          className="flex h-13 shrink-0 items-center justify-center gap-2 rounded-[1rem] bg-red-800 px-4 text-xs font-black tracking-[0.08em] text-white shadow-button transition-[background-color,box-shadow,transform] duration-300 hover:-translate-y-0.5 hover:bg-red-700 hover:shadow-[0_12px_28px_rgba(220,38,38,0.32)] active:translate-y-0 disabled:cursor-wait disabled:opacity-75 disabled:hover:translate-y-0 sm:px-6 sm:text-sm"
        >
          {isLoading ? (
            <LoaderCircle className="h-5 w-5 animate-spin" strokeWidth={2} aria-hidden="true" />
          ) : (
            <Sparkles className="h-4.5 w-4.5" strokeWidth={2} aria-hidden="true" />
          )}
          <span className="hidden sm:inline">SORGULA</span>
        </button>
      </form>

      <SuggestedQueries onSelect={selectSuggestion} disabled={isLoading} />
      <p className="sr-only" aria-live="polite">
        {announcement}
      </p>
    </div>
  );
}
