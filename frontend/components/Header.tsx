import { HeartPulse } from "lucide-react";

function BrandMark() {
  return (
    <span
      className="brand-mark grid h-11 w-11 place-items-center text-[#7a1014]"
      aria-hidden="true"
    >
      <HeartPulse className="h-10 w-10" strokeWidth={2.5} />
    </span>
  );
}

export function Header() {
  return (
    <header className="relative z-50 mx-auto flex w-full max-w-[1440px] items-center justify-between px-5 py-5 sm:px-8 lg:px-12 lg:py-7">
      <a href="#top" className="group flex items-center gap-3" aria-label="Şifacı AI ana sayfa">
        <BrandMark />
        <span className="brand-wordmark text-[1.7rem] font-black tracking-[-0.055em] text-stone-950 sm:text-[2rem]">
          ŞİFACI <span className="font-normal text-[#7a1014]">AI</span>
        </span>
      </a>

      <nav className="hidden items-center gap-7 text-sm font-medium text-muted lg:flex" aria-label="Ana menü">
        <a className="transition-colors hover:text-medical-deep" href="#capabilities">
          Neler sorabilirsiniz?
        </a>
        <a className="transition-colors hover:text-medical-deep" href="#safety">
          Güvenlik
        </a>
      </nav>
    </header>
  );
}
