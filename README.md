# Şifacı

Şifacı, kullanıcının eklediği ilaç belgelerini yerel olarak indeksleyen ve
yalnızca bu kayıtları kaynak göstererek cevap veren yerel bir RAG
uygulamasıdır. Next.js arayüzü FastAPI üzerinden, Streamlit arayüzü ise
doğrudan Python katmanından aynı güvenli `answer_query()` akışını kullanır.
Kişisel reçete, tanı, doz hesabı veya tedavi önerisi vermez.

Soru yanıtlama sırasında internet veya API anahtarı gerekmez. İlaç çözümleme,
FTS5/embedding retrieval ve cevap üretimi yerel SQLite verisi ile Microsoft
Foundry Local modelleri üzerinde çalışır.

## Hızlı başlatma

İlk kullanımda D: depolamasını ve sanal ortamı hazırlayın:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_d_drive.ps1
```

Windows'ta proje klasöründe aşağıdaki tek komut hem Python API'yi hem de
Next.js arayüzünü başlatır ve tarayıcıyı açar:

```powershell
.\start_sifaci.cmd
```

Tarayıcıyı otomatik açmadan başlatmak için:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_sifaci.ps1 -NoBrowser
```

Başlangıç hataları olursa ayrıntılı loglar `D:\SifaciAI\runtime` klasörüne
yazılır. Başlatıcı her çalışmada `D:\SifaciAI\data\medicines` altındaki JSON
dosyalarını kanonik `D:\SifaciAI\data\medicines.db` veritabanıyla eşitler;
boş veritabanıyla sessizce başlamaz.

Veri zincirini model çağırmadan kontrol etmek için:

```powershell
.\scripts\run_python.ps1 -m scripts.diagnose_database --skip-retrieval
```

Gerçekten yanıtlanabilir ilaç sayısını denetlemek ve MVP eşiğine yalnızca
gerektiği kadar veri işleyerek ulaşmak için:

```powershell
.\scripts\run_python.ps1 -m scripts.prepare_mvp_dataset --target-ready 200
.\scripts\run_python.ps1 -m scripts.validate_mvp
```

`READY`, yalnızca katalogda bulunmak demek değildir. Belgesi indirilmiş ve
ayrıştırılmış, chunk'ları aynı kanonik ilaca bağlı, güncel yerel embeddingleri
ve FTS5 kayıtları eksiksiz ilaçları ifade eder. `/health` yanıtı bu nedenle
`catalog_medicine_count` ile `ready_medicine_count` değerlerini ayrı bildirir.

Eski KÜB/KT adlandırmalarının oluşturduğu olası çift kayıtlar önce salt-okunur
olarak incelenebilir. Belirsiz eşleşmeler otomatik birleştirilmez:

```powershell
.\scripts\run_python.ps1 -m scripts.repair_medicine_links
# Rapor onaylandıktan sonra, SQLite yedeği oluşturarak uygular:
.\scripts\run_python.ps1 -m scripts.repair_medicine_links --apply
```

Embedding ve retrieval dahil tam kontrol için:

```powershell
.\scripts\run_python.ps1 -m scripts.diagnose_database --query "Parol'un yan etkileri nelerdir?"
```

## Veri akışı

```text
D:\SifaciAI\data\medicines\*.{json,csv}
  -> src.ingestion (doğrulama ve güvenli chunking)
  -> SQLite medicines + document_chunks + FTS5
  -> ilaç/etkin madde çözümleme
  -> FTS5 (ve mevcutsa embedding) ile en alakalı KÜB/KT parçaları
  -> güvenlik sınırlı RAG context
  -> Microsoft Foundry Local chat modeli
  -> FastAPI POST /api/chat / Streamlit
  -> Şifacı cevap kartı ve kaynak listesi
```

Embedding vektörleri SQLite içinde JSON dizileri olarak saklanır. Her chunk,
üretildiği embedding modelinin adını da taşır; model ayarı değişirse ingestion
eski vektörleri yeniden üretir. Aynı, değişmemiş belge tekrar işlendiğinde chunk
kayıtları çoğaltılmaz.

Embedding aşaması başlarken Foundry Local modeli bir kez yüklenir ve işlem
bitene kadar RAM'de tutulur. Chunk'lar `EMBEDDING_BATCH_SIZE=32` (isteğe bağlı
`64`) gruplarıyla modele gönderilir. Her batch sonucu merkezi SQLite yazıcısına
tek `executemany` transaction'ı olarak kaydedilir ve `tqdm` ilerleme çubuğu
tamamlanan chunk sayısını gösterir. Kesinti olursa yazılmamış kayıtlar `PENDING`
kalır; tamamlanmış batch'ler yeniden üretilmez.

## Proje yapısı

```text
app.py                         Streamlit chat arayüzü
api.py                         Next.js için yerel FastAPI katmanı
frontend/                      Next.js premium Şifacı arayüzü
pages/1_İlaç_Ekle.py           Yönetici ilaç ekleme formu
config.py                      Tüm çalışma ayarları
D:\SifaciAI\data\medicines\  Çalışma zamanı ilaç belgeleri
src/database.py                SQLite şeması ve veri erişimi
src/ingestion.py               JSON okuma, chunking ve embedding
src/embeddings.py              Foundry Local embedding istemcisi
src/retrieval.py               Cosine similarity ve isim önceliği
src/rag.py                     Güvenlik sınırlı retrieval-augmented cevap
src/medicine_service.py        İlaç/etkin madde arama ve ilgili chunk servisi
src/foundry_runtime.py         Paylaşılan Foundry Local SDK başlatma katmanı
src/foundry_client.py          Foundry Local chat istemcisi
scripts/debug_medicine.py      İlaç/doküman/chunk/embedding durum tabloları
scripts/debug_query.py         Intent, eşleşme, cosine skor ve LLM context izi
scripts/data_coverage.py       Katalog kapsama sayıları ve yüzdeleri
tests/                         Birim, güvenlik ve uçtan uca akış testleri
```

## Chat API

Backend'i doğrudan başlatmak ve RAG endpoint'ini çağırmak için:

```powershell
.\scripts\run_python.ps1 api.py

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/chat `
  -ContentType "application/json" `
  -Body '{"message":"Parolun yan etkileri nelerdir?"}'
```

Yanıt `answer`, tespit edilen `medicine`, `sources`, `disclaimer` ve gerektiğinde
ürün formu `suggestions` alanlarını içerir.

İzole örnek veritabanı ve testler:

```powershell
.\scripts\run_python.ps1 -m scripts.seed_medicines
.\scripts\run_python.ps1 -m pytest -q
```

## JSON ve CSV formatı

`data/medicines/example_medicine.json` şablondur ve `"is_example": true`
olduğu için ingestion sırasında atlanır. Kendi dosyanızda bu alanı kaldırın
veya `false` yapın. Boş alanlar uydurulmaz; boş olarak saklanabilir.

Toplu CSV şablonu `data/examples/medicines_bulk_template.csv` dosyasındadır.
Hazırlanan `.json` ve `.csv` dosyaları `D:\SifaciAI\data\medicines` altına
konduktan sonra:

```powershell
.\scripts\run_python.ps1 -m src.ingestion
.\scripts\run_python.ps1 scripts/verify_medicine.py SELECTRA
```

Ingestion; yeni, güncellenen, duplicate ve hatalı kayıt sayılarını ayrı ayrı
gösterir. Aynı ürün adı tekrar geldiğinde upsert yapılır; değişmemiş chunk ve
embeddingler yeniden oluşturulmaz.

## Otomatik TİTCK senkronizasyonu

Birincil kaynaklar TİTCK'ın Ruhsatlı Beşeri Tıbbi Ürünler Listesi ile KÜB/KT
sistemidir. İlk aktarım internet ve yeterli disk alanı gerektirir. Tam aktarım
sonrasında PDF metinleri, SQLite metadata ve embeddingler yerelde tutulur.

İlk kurulum:

```powershell
.\scripts\run_python.ps1 -m pip install -r requirements.txt
.\scripts\run_python.ps1 scripts/sync_titck.py --bootstrap
```

Güncelleme ve yarıda kalan aktarımı sürdürme:

```powershell
.\scripts\run_python.ps1 scripts/sync_titck.py --update
.\scripts\run_python.ps1 scripts/sync_titck.py --resume
.\scripts\run_python.ps1 scripts/sync_titck.py --resume --medicine "MUSCOFLEX"
```

Uzun tam aktarımı terminale bağlı kalmadan arka planda başlatmak ve durumunu
görmek için:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_bulk_sync.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\bulk_sync_status.ps1
.\scripts\run_python.ps1 scripts\pipeline_status.py
```

Aktarım aşamalı ve bounded queue tabanlıdır: `DISCOVERY → DOWNLOAD → PARSE →
CHUNK → EMBED → DB WRITE`. Varsayılan olarak 6 kalıcı HTTP oturumu indirme,
4 işçi PDF ayrıştırma yapar. Embedding modeli bir kez yüklenir; sonuçlar tek
SQLite yazıcısı tarafından 500 olaylık transaction gruplarıyla WAL modunda
kaydedilir. URL discovery sonucu veritabanında saklanır ve yalnızca
`--update` ile kontrollü olarak yenilenir.

`SQLiteConnectionManager`, her veritabanı yolu için tek thread-safe yazıcı
yönetir. Yazmalar `BEGIN IMMEDIATE` transaction'larıyla sıraya alınır; okumalar
WAL üzerinde bağımsız query-only bağlantılarla devam eder. Her bağlantıda
`busy_timeout=30000`, `synchronous=NORMAL` ve foreign key denetimi uygulanır.
Toplu ürün, chunk ve pipeline kayıtları 100 veya varsayılan 500 satırlık
`executemany`/batch gruplarıyla işlenir.

100 bekleyen belge üzerinde performans ölçümü:

```powershell
.\scripts\run_python.ps1 scripts\benchmark_pipeline.py --limit 100
```

Temel ayarlar `config.py` içinde merkezidir ve ortam değişkenleriyle
değiştirilebilir:

```powershell
$env:DOWNLOAD_WORKERS = "6"
$env:PARSE_WORKERS = "4"
$env:DB_WRITE_BATCH_SIZE = "500"
$env:CONNECT_TIMEOUT = "10"
$env:READ_TIMEOUT = "30"
$env:MAX_RETRIES = "5"
$env:EMBEDDING_BATCH_SIZE = "32"
$env:OCR_ENABLED = "true"
$env:OCR_LANGUAGE = "tur"
$env:OCR_DPI = "150"
$env:TESSERACT_CMD = "D:\SifaciAI\tools\Tesseract-OCR\tesseract.exe"
.\scripts\run_python.ps1 scripts\sync_titck.py --resume
```

Bağlantı/DNS/timeout ile HTTP 429, 500, 502, 503 ve 504 hataları `tenacity`
exponential backoff politikasıyla en fazla 5 kez denenir. HTTP 404 gibi kalıcı
hatalar `FAILED_PERMANENT` olarak kaydedilir ve tekrar indirme kuyruğuna
alınmaz. `Ctrl+C` işlemi güvenle duraklatır; sonraki `--resume` yalnızca
`PENDING` ve `FAILED_RETRYABLE` dokümanları sürdürür.

Belge pipeline'ı varsayılan olarak 6 paralel indirme ve 4 paralel parse
worker'ı kullanır. `DOWNLOAD_WORKERS` 4–8, `PARSE_WORKERS` 2–8 aralığında
ayarlanabilir. Aşamalar bounded queue'larla birbirinden ayrılır; tüm SQLite
yazmaları tek writer thread'i tarafından batch transaction olarak yapılır.
Checkpoint her 50 ilerlemede, en geç 30 saniyede bir ve pipeline kapanırken
yazılır. Bilgisayar uyku/askı modundan döndüğünde süreç kaldığı yerden
devam eder; zorunlu kapanmada `--resume` kalıcı durum kayıtlarını kullanır.

Windows çalışma verileri zorunlu olarak `D:\SifaciAI` altında tutulur; C: için
otomatik geri dönüş yoktur. Senkronizasyon, veri diskinin boş alanı güvenlik
eşiğine ulaştığında durur ve `--resume` ile kaldığı yerden sürdürülebilir.

Geliştirme sırasında az sayıda kayıtla kaynak bağlantısını sınamak için
`--limit 10` kullanılabilir. Tam veri seti doğrulaması ve şeffaf RAG testi:

```powershell
.\scripts\run_python.ps1 scripts/verify_database.py
.\scripts\run_python.ps1 scripts/verify_database.py --medicine "selectra"
.\scripts\run_python.ps1 scripts/test_rag.py "selectra yan etkileri neler"
.\scripts\run_python.ps1 scripts/debug_medicine.py "MUSCOFLEX"
.\scripts\run_python.ps1 scripts/debug_query.py "Muscoflex yan etkileri neler?"
.\scripts\run_python.ps1 scripts/debug_query.py "Muscoflex yan etkileri neler?" --call-llm
.\scripts\run_python.ps1 scripts/data_coverage.py
```

`debug_query.py` varsayılan olarak LLM çağrısını simüle eder; ilaç tespiti,
intent ve gerçek cosine retrieval yine çalışır. Yerel modele de istek göndermek
için `--call-llm` kullanılır.

Senkronizasyon şu zinciri otomatik yürütür: güncel XLSX bağlantısını bulma,
kolon normalizasyonu, ürün ve alias upsert'i, KÜB/KT URL toplama, hash kontrollü
PDF indirme, PyMuPDF metin çıkarma, başlık bazlı bölümleme ve 32'li local
embedding batch'leri. TİTCK dışı belge URL'leri güvenlik amacıyla reddedilir.
Metin katmanı olmayan PDF sayfaları 150 DPI görüntüye dönüştürülerek Tesseract
Türkçe OCR fallback'ine gönderilir. Python sarmalayıcısı D sürücüsündeki sanal
ortamda, OCR motoru ve `tur.traineddata` ise varsayılan olarak
`D:\SifaciAI\tools\Tesseract-OCR` altında tutulur. OCR çalışma zamanı geçici
olarak kullanılamazsa belge `FAILED_RETRYABLE` kalır; önceki OCR kaynaklı kalıcı
hatalar ilk `--resume` çalışmasında otomatik olarak yeniden kuyruğa alınır.

## İlacabak ikincil veri sağlayıcısı

İlacabak entegrasyonu mevcut TİTCK verilerini silmez veya değiştirmez. Kaynak
önceliği `TİTCK KÜB/KT → üretici resmi belge → İlacabak prospektüs HTML`
şeklindedir. Ürün eşleşmesi ve HKT/KÜB bağlantıları kaydedilir; erişilebilen
prospektüs HTML'i başlıklara göre ayrıştırılır. İlacabak `robots.txt` dosyasının
engellediği `/pdf/` yolları indirilmez ve erişim engeli aşılmaya çalışılmaz.

Önce ek tabloları güvenli, eklemeli migration ile oluşturun:

```powershell
.\scripts\run_python.ps1 scripts\migrate_ilacabak.py
```

Küçük bir deneme, yarım kalan aktarımı sürdürme ve kontrollü güncelleme:

```powershell
.\scripts\run_python.ps1 scripts\sync_ilacabak.py --resume --limit 10
.\scripts\run_python.ps1 scripts\sync_ilacabak.py --resume --medicine "SELECTRA 50" --limit 1
.\scripts\run_python.ps1 scripts\sync_ilacabak.py --resume
.\scripts\run_python.ps1 scripts\sync_ilacabak.py --update
.\scripts\run_python.ps1 scripts\verify_ilacabak_selectra.py
```

Varsayılan ayarlar `config.py` içindedir: 3 işçi, site genelinde istekler
arasında 0,5 saniye bekleme ve 4 yeniden deneme. HTML içerikleri
`D:\SifaciAI\cache\ilacabak` altında URL hash'iyle saklanır. Aynı içerik hash'i tekrar
görülürse embedding oluşturulmaz. Yaş/doz ifadeleri LLM ile tahmin edilmez;
yalnızca kaynakta açıkça yazan yaş sınırı ve sıklık metni
`medicine_dosage_rules` tablosuna alınır.

## Windows üzerinde sıfırdan kurulum

PowerShell açın:

```powershell
git clone https://github.com/suaslan/sifaci.git
cd sifaci

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_d_drive.ps1
.\scripts\run_python.ps1 -m pytest -q tests
.\scripts\run_python.ps1 main.py
```

İlaç JSON dosyanızı hazırlayıp aktarın:

```powershell
Copy-Item .\data\medicines\example_medicine.json D:\SifaciAI\data\medicines\my_medicine.json
notepad D:\SifaciAI\data\medicines\my_medicine.json
.\scripts\run_python.ps1 -m src.ingestion
```

Editörde örnek değerleri gerçek, doğrulanmış ürün bilgileriyle değiştirin ve
`is_example` alanını kaldırın ya da `false` yapın.

Foundry Local modellerini ayrı ayrı doğrulamak ve uygulamayı açmak için:

```powershell
.\scripts\run_python.ps1 -c "from src.embeddings import generate_embedding; print('Embedding boyutu:', len(generate_embedding('deneme metni')))"
.\scripts\run_python.ps1 -m src.foundry_client
.\scripts\run_python.ps1 -m scripts.smoke_pipeline
.\scripts\run_python.ps1 -m streamlit run app.py
```

Next.js arayüzünü kullanmak için iki PowerShell terminali açın. İlk terminal:

```powershell
.\scripts\run_python.ps1 api.py
```

İkinci terminal:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

Arayüz `http://localhost:3000`, API sağlık kontrolü ise
`http://127.0.0.1:8000/health` adresindedir. Python API farklı bir adreste
çalışacaksa `frontend/.env.local` içinde `SIFACI_BACKEND_URL` ayarlanabilir.

İlk model çağrısında yürütme sağlayıcıları ve model dosyaları indirilebilir.

## Merkezi ayarlar

Varsayılanlar `config.py` ile `scripts/storage.ps1` içindedir. SQLite, indirilen
dokümanlar, çalışma logları, geçici dosyalar, Python bytecode, pip/npm/pytest,
HuggingFace/Transformers/Torch, Playwright ve Next.js cache'leri ile Foundry
Local model dosyaları `D:\SifaciAI` altında tutulur. Aşağıdaki çalışma ayarları
ortam değişkenleriyle değiştirilebilir:

- `DEBUG=true`
- `FOUNDRY_APP_NAME=sifaci`
- `FOUNDRY_MODEL_ALIAS=qwen2.5-0.5b`
- `EMBEDDING_MODEL_NAME=qwen3-embedding-0.6b`
- `EMBEDDING_DEVICE_TYPE=CPU` (GPU/WebGPU uyumsuzluğunda güvenli varsayılan; `AUTO`, `GPU` veya `NPU` seçilebilir)
- `FOUNDRY_PREPARE_EXECUTION_PROVIDERS=true`
- `FOUNDRY_APP_DATA_DIR=D:\SifaciAI\foundry`
- `FOUNDRY_MODEL_CACHE_DIR=D:\SifaciAI\models`
- `FOUNDRY_LOGS_DIR=D:\SifaciAI\logs`
- `CHAT_TEMPERATURE=0.0`
- `CHAT_MAX_TOKENS=600`
- `MAX_CHUNK_CHARS=1200`
- `RETRIEVAL_TOP_K=5`
- `MINIMUM_SIMILARITY_SCORE=0.35`
- `MEDICINE_NAME_BOOST=0.25`
- `MAX_QUESTION_CHARS=1000`
- `API_HOST=127.0.0.1`
- `API_PORT=8000`
- `SIFACI_STORAGE_ROOT=D:\SifaciAI`
- `SYNC_MIN_FREE_GB=0.75`
- `EMBEDDING_BATCH_SIZE=32`

Debug arayüzünü açmak için Streamlit'ten önce:

```powershell
$env:DEBUG = "true"
.\scripts\run_python.ps1 -m streamlit run app.py
```

## Güvenlik sınırları

- Model yalnızca retrieval context içindeki metni kullanmalıdır.
- Kaynakta olmayan doz, sıklık, süre veya uygulama yolu oluşturulamaz.
- Kişiye özel doz ve tedavi soruları backend tarafından model çağrısından önce
  engellenir.
- Ciddi yan etki kayıtları görünür güvenlik uyarısı üretir.
- Düşük benzerlikte veya eksik bilgide sabit “bilgi bulunmuyor” cevabı verilir.
- Her cevap kaynak adı ve kişisel tıbbi değerlendirme uyarısıyla tamamlanır.
