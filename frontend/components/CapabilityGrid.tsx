const capabilities = [
  { number: "01", title: "Kullanım bilgileri", detail: "Kayıtlı ürün talimatları" },
  { number: "02", title: "Yan etkiler", detail: "Yaygın ve ciddi etkiler" },
  { number: "03", title: "Etken maddeler", detail: "Doğrulanmış ürün içeriği" },
  { number: "04", title: "Uyarılar", detail: "Kontrendikasyon ve güvenlik" },
] as const;

export function CapabilityGrid() {
  return (
    <section id="capabilities" className="mx-auto w-full max-w-[1440px] px-5 py-20 sm:px-8 lg:px-12 lg:py-28">
      <div className="mb-9 flex items-end justify-between gap-6 border-b border-line pb-5">
        <div>
          <p className="mb-2 text-[0.68rem] font-semibold tracking-[0.2em] text-medical-deep uppercase">
            Bilgi alanları
          </p>
          <h2 className="text-2xl font-medium tracking-[-0.025em] text-ink sm:text-3xl">
            Ne sorabilirsiniz?
          </h2>
        </div>
        <p className="hidden max-w-sm text-right text-sm leading-6 text-muted md:block">
          Yalnızca sisteme eklenmiş, kaynaklandırılmış ürün bilgilerinde arama yapılır.
        </p>
      </div>

      <div className="grid gap-px overflow-hidden rounded-[1.35rem] border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
        {capabilities.map((capability) => (
          <article
            key={capability.number}
            className="group min-h-48 bg-canvas p-6 transition-[background-color,transform] duration-300 hover:-translate-y-1 hover:bg-paper sm:p-7"
          >
            <span className="text-xs font-semibold tracking-[0.16em] text-medical-deep">
              {capability.number}
            </span>
            <div className="mt-16">
              <h3 className="text-lg font-medium tracking-[-0.02em] text-ink">
                {capability.title}
              </h3>
              <p className="mt-1.5 text-sm text-muted">{capability.detail}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
