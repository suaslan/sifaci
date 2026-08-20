"""Uygulamanin ortak ayarlari.

Bu dosyada gizli anahtar tutulmaz. Foundry Local ve embedding modeli yerel
calisacagi icin uygulamanin bir bulut API anahtarina ihtiyaci olmayacaktir.
"""

import os
from pathlib import Path


def _environment_flag(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment variable."""

    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Uygulama bilgisi -----------------------------------------------------------

APP_NAME = "Şifacı"

# Geliştirme sırasında retrieval ayrıntılarını arayüzde gösterir. Production
# ortamında varsayılan olarak kapalıdır.
DEBUG = _environment_flag("DEBUG", default=False)


# Proje klasorleri -----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MEDICINE_DOCUMENTS_DIR = DATA_DIR / "medicines"
DATABASE_PATH = DATA_DIR / "medicines.db"


# Yerel modeller -------------------------------------------------------------

# Degerler, kodu degistirmeden ortam degiskenleriyle ezilebilir.
FOUNDRY_MODEL_ALIAS = os.getenv("FOUNDRY_MODEL_ALIAS", "qwen2.5-0.5b")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "qwen3-0.6b-embedding",
)


# Retrieval ayarlari ---------------------------------------------------------

# Kullanici sorusuna en yakin kac kaydin getirilecegi.
RETRIEVAL_TOP_K = 5

# Bu deger ilk testlerden sonra veri setine gore yeniden ayarlanacaktir.
MINIMUM_SIMILARITY_SCORE = 0.35


# Guvenli cevap --------------------------------------------------------------

MISSING_INFORMATION_RESPONSE = "Bu bilgi mevcut ilaç veri tabanında bulunmuyor."
