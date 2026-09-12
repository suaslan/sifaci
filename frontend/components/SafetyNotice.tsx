import { ShieldCheck } from "lucide-react";

export function SafetyNotice() {
  return (
    <footer id="safety" className="border-t border-line/80 px-5 py-8 sm:px-8 lg:px-12">
      <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-5 text-muted md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-3 text-xs font-semibold tracking-[0.16em] uppercase">
          <ShieldCheck className="h-4 w-4 text-medical-deep" strokeWidth={1.7} aria-hidden="true" />
          Şifacı AI / Yerel ilaç bilgi asistanı
        </div>
        <p className="max-w-3xl text-xs leading-5 md:text-right">
          Bu sistem yalnızca kayıtlı ilaç bilgilerinin görüntülenmesi amacıyla hazırlanmıştır.
          Kişisel tıbbi değerlendirme veya tedavi önerisi yerine geçmez.
        </p>
      </div>
    </footer>
  );
}
