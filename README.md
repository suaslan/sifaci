# Şifacı

Şifacı, kullanıcının eklediği ilaç belgelerini yerel olarak indeksleyen ve
yalnızca bu kayıtları kaynak göstererek cevap veren bir Streamlit RAG
uygulamasıdır. Kişisel reçete, tanı, doz hesabı veya tedavi önerisi vermez.

## Veri akışı

```text
data/medicines/*.json
  -> src.ingestion (doğrulama ve güvenli chunking)
  -> Microsoft Foundry Local embedding modeli
  -> SQLite medicines + document_chunks
  -> soru embedding'i + cosine similarity
  -> en alakalı chunk'lar
  -> güvenlik sınırlı RAG context
  -> Microsoft Foundry Local chat modeli
  -> Streamlit chat cevabı ve kaynak listesi
```

Embedding vektörleri SQLite içinde JSON dizileri olarak saklanır. Her chunk,
üretildiği embedding modelinin adını da taşır; model ayarı değişirse ingestion
eski vektörleri yeniden üretir. Aynı, değişmemiş belge tekrar işlendiğinde chunk
kayıtları çoğaltılmaz.

## Proje yapısı

```text
app.py                         Streamlit chat arayüzü
pages/1_İlaç_Ekle.py           Yönetici ilaç ekleme formu
config.py                      Tüm çalışma ayarları
data/medicines/                Kullanıcı JSON belgeleri
src/database.py                SQLite şeması ve veri erişimi
src/ingestion.py               JSON okuma, chunking ve embedding
src/embeddings.py              Foundry Local embedding istemcisi
src/retrieval.py               Cosine similarity ve isim önceliği
src/rag.py                     Güvenlik sınırlı retrieval-augmented cevap
src/foundry_runtime.py         Paylaşılan Foundry Local SDK başlatma katmanı
src/foundry_client.py          Foundry Local chat istemcisi
tests/                         Birim, güvenlik ve uçtan uca akış testleri
```

## JSON formatı

`data/medicines/example_medicine.json` şablondur ve `"is_example": true`
olduğu için ingestion sırasında atlanır. Kendi dosyanızda bu alanı kaldırın
veya `false` yapın. Boş alanlar uydurulmaz; boş olarak saklanabilir.

## Windows üzerinde sıfırdan kurulum

PowerShell açın:

```powershell
git clone https://github.com/suaslan/sifaci.git
cd sifaci

py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q tests
python main.py
```

İlaç JSON dosyanızı hazırlayıp aktarın:

```powershell
Copy-Item .\data\medicines\example_medicine.json .\data\medicines\my_medicine.json
notepad .\data\medicines\my_medicine.json
python -m src.ingestion
```

Editörde örnek değerleri gerçek, doğrulanmış ürün bilgileriyle değiştirin ve
`is_example` alanını kaldırın ya da `false` yapın.

Foundry Local modellerini ayrı ayrı doğrulamak ve uygulamayı açmak için:

```powershell
python -c "from src.embeddings import generate_embedding; print('Embedding boyutu:', len(generate_embedding('deneme metni')))"
python -m src.foundry_client
python -m scripts.smoke_pipeline
streamlit run app.py
```

İlk model çağrısında yürütme sağlayıcıları ve model dosyaları indirilebilir.

## Merkezi ayarlar

Varsayılanlar `config.py` içindedir. Aşağıdaki ortam değişkenleri kodu
değiştirmeden kullanılabilir:

- `DEBUG=true`
- `FOUNDRY_APP_NAME=sifaci`
- `FOUNDRY_MODEL_ALIAS=qwen2.5-0.5b`
- `EMBEDDING_MODEL_NAME=qwen3-embedding-0.6b`
- `FOUNDRY_PREPARE_EXECUTION_PROVIDERS=true`
- `CHAT_TEMPERATURE=0.0`
- `CHAT_MAX_TOKENS=600`
- `MAX_CHUNK_CHARS=1200`
- `RETRIEVAL_TOP_K=5`
- `MINIMUM_SIMILARITY_SCORE=0.35`
- `MEDICINE_NAME_BOOST=0.12`
- `MAX_QUESTION_CHARS=1000`

Debug arayüzünü açmak için Streamlit'ten önce:

```powershell
$env:DEBUG = "true"
streamlit run app.py
```

## Güvenlik sınırları

- Model yalnızca retrieval context içindeki metni kullanmalıdır.
- Kaynakta olmayan doz, sıklık, süre veya uygulama yolu oluşturulamaz.
- Kişiye özel doz ve tedavi soruları backend tarafından model çağrısından önce
  engellenir.
- Ciddi yan etki kayıtları görünür güvenlik uyarısı üretir.
- Düşük benzerlikte veya eksik bilgide sabit “bilgi bulunmuyor” cevabı verilir.
- Her cevap kaynak adı ve kişisel tıbbi değerlendirme uyarısıyla tamamlanır.
