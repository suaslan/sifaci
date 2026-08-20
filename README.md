# Şifacı

Bu proje, yalnizca kullanicinin sisteme ekledigi ilac belgelerinden cevap
uretecek yerel bir RAG uygulamasinin baslangic iskeletidir. Bir receteleme
sistemi degildir.

## Proje yapisi

```text
şifacı/
|-- app.py
|-- main.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- medicines/
|   `-- medicines.db
|-- src/
|   |-- __init__.py
|   |-- database.py
|   |-- embeddings.py
|   |-- ingestion.py
|   |-- retrieval.py
|   |-- rag.py
|   `-- foundry_client.py
`-- tests/
```

## Dosyalarin gorevleri

- `app.py`: Streamlit kullanici arayuzunun giris noktasi olacak.
- `main.py`: Terminalden calistirma ve gelistirme kontrolleri icin basit giris
  noktasi.
- `config.py`: Dosya yollari, yerel model adlari, retrieval ayarlari ve guvenli
  varsayilan cevap gibi ortak ayarlari tutar.
- `requirements.txt`: Projenin Python kutuphanelerini listeler. `sqlite3`,
  Python ile birlikte geldigi icin ayrica eklenmez.
- `data/medicines/`: Sisteme yuklenecek kaynak ilac belgelerinin yeri.
- `data/medicines.db`: Ilac metinleri ve embedding vektorleri icin kullanilacak
  SQLite veritabani. Tablo semasi sonraki asamada olusturulacak.
- `src/database.py`: SQLite baglantisi ve veri erisim islemleri.
- `src/embeddings.py`: Yerel embedding modelinin yuklenmesi ve vektor uretimi.
- `src/ingestion.py`: Belgelerin okunmasi, parcalanmasi ve veritabanina
  aktarilmasi.
- `src/retrieval.py`: Cosine similarity tabanli vektor aramasi.
- `src/rag.py`: Retrieval, guvenli prompt ve cevap uretme akisinin birlestigi
  modul.
- `src/foundry_client.py`: Foundry Local SDK entegrasyonunun uygulamanin geri
  kalanindan ayrildigi modul.
- `tests/`: Otomatik testler bu klasore eklenecek.

## Kurulum

Python 3.11 veya daha yeni bir surum kullanin.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Basit giris dosyasini calistirmak icin:

```powershell
python main.py
```

Foundry Local bağlantısını ve yerel modeli test etmek için:

```powershell
python -m src.foundry_client
```

İlk test sırasında seçilen model ve Windows ML yürütme sağlayıcısı indirilebilir.
İndirme tamamlandıktan sonra prompt işleme cihazdaki Foundry Local çalışma zamanı
üzerinde gerçekleşir. Kullanılan model adı yalnızca `config.py` dosyasındaki
`FOUNDRY_MODEL_ALIAS` ayarından değiştirilir.

## Guvenlik sinirlari

Uygulama tamamlandiginda model yalnizca retrieval sonucunda bulunan kaynaklari
kullanacak; doz, kullanim sikligi veya yan etki bilgisi uretmeyecek ya da tahmin
etmeyecektir. Kaynaklarda cevap yoksa su sabit yanit verilecektir:

> Bu bilgi mevcut ilac veri tabaninda bulunmuyor.

Bu ilk asamada veritabani semasi, belge aktarimi, vektor aramasi, Foundry Local
cagrisi ve Streamlit arayuzu bilerek uygulanmamistir.
