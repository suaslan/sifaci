"""Şifacı uygulamasının basit geliştirme giriş noktası.

RAG akisinin parcalari sonraki asamalarda ``src`` modullerinden cagrilacak.
Bu ilk surum yalnizca proje ayarlarinin okunabildigini gosterir.
"""

from config import APP_NAME, DATABASE_PATH, FOUNDRY_MODEL_ALIAS


def main() -> None:
    """Baslangic bilgilerini terminale yazdir."""

    print(f"{APP_NAME} - Yerel İlaç Bilgi Asistanı")
    print(f"Veritabani: {DATABASE_PATH}")
    print(f"Foundry Local modeli: {FOUNDRY_MODEL_ALIAS}")
    print("RAG islevleri sonraki asamalarda eklenecek.")


if __name__ == "__main__":
    main()
