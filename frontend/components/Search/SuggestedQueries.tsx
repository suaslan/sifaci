const suggestions = [
  "Parol günde kaç kez alınır?",
  "Muscoflex kas gevşetir mi?",
  "Arveles aç karnına mı içilir?",
] as const;

type SuggestedQueriesProps = {
  onSelect: (query: string) => void;
  disabled?: boolean;
};

export function SuggestedQueries({ onSelect, disabled = false }: SuggestedQueriesProps) {
  return (
    <div className="flex max-w-full flex-wrap justify-center gap-2.5" aria-label="Örnek sorgular">
      {suggestions.map((suggestion) => (
        <button
          key={suggestion}
          type="button"
          disabled={disabled}
          onClick={() => onSelect(suggestion)}
          className="rounded-full border border-red-800/25 bg-white/70 px-4 py-2 text-xs font-semibold text-stone-700 shadow-[0_5px_18px_rgba(153,27,27,0.05)] backdrop-blur-md transition-[border-color,background-color,color,transform,box-shadow] duration-300 hover:-translate-y-0.5 hover:border-red-700/55 hover:bg-white hover:text-red-900 hover:shadow-[0_8px_22px_rgba(153,27,27,0.10)] disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0"
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}
