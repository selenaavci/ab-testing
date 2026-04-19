🧪 A/B Testing Agent

📌 Overview
A/B Testing Agent, marketing, product ve growth ekiplerinin istatistik uzmanı olmadan A/B test tasarlayabileceği, veri yükleyip saniyeler içinde net bir "yayına al / alma / devam et" kararı alabileceği bir deney yönetim aracıdır.
Tüm istatistiksel analizler, validasyonlar ve karar gerekçeleri arka planda otomatik çalışır; kullanıcı yalnızca üç adımda ilerler:

1. Deneyimi tanımla (ne test ediyorsun, başarıyı nasıl ölçüyorsun)
2. Veriyi yükle (kolonlar otomatik algılanır)
3. Sonucu gör (tek karar kartı + isteğe bağlı teknik detaylar)

🎯 Project Purpose
Kurum içinde deney kültürünü yaygınlaştırmak
Business ekiplerin teknik destek almadan A/B test yapabilmesini sağlamak
Yanlış test kurgularını ve hatalı yorumlamaları engellemek
Karar alma süreçlerini hızlandırmak ve veri odaklı hale getirmek
Teknik bilgisi sınırlı kullanıcılarda da güven seviyesini yüksek tutmak

👥 Target Users
Marketing & Growth Teams
Product Managers
Digital Analytics Teams
CRM & Campaign Teams
Data Analysts (detay istediklerinde Advanced expander'ları üzerinden)

🧭 User Experience — 3 adımlı sihirbaz

1. Deneyim
- Hipotez (serbest metin)
- "Başarıyı nasıl ölçüyorsun?" → Dönüşüm / Evet-Hayır ya da Gelir / Süre / Sayısal değer
- "En az ne kadarlık bir iyileşme anlamlı?" slider'ı (MDE yerine günlük dil)
- Gelişmiş ayarlar expander'ı: güven seviyesi (%90/95/99) ve güç (power)

2. Veri
- Tek tıkla CSV/XLSX yükleme
- Varyant, metric, zaman, segment ve önceki dönem kolonları otomatik algılanır
- "Manuel ayarla" expander'ı ile elle düzeltme
- Veri önizleme

3. Sonuç
- Renkli karar kartı:
  * Varyant kazandı. Yayına alınabilir.
  * Varyant yayına alınmamalı.
  * Kesin karar için teste devam edilmeli.
  * Kontrol ile varyant arasında anlamlı fark yok.
  * Testi yeniden kurgulayın.
- Kontrol / varyant rakamları + relatif fark + "Güven: Yüksek/Yeterli/Düşük"
- Örneklem doluluk çubuğu (mevcut / hedef)
- Karar gerekçeleri (tek cümlelik açıklamalar)
- Karşılaştırma grafiği ve (varsa) zamansal trend
- İsteğe bağlı Detaylı expander'lar:
  * Veri kalitesi (SRM, duplicate, missing, outlier)
  * Yan metrikler (guardrail)
  * Segment bazında performans + FDR düzeltmesi
  * CUPED varyans azaltma
  * Bayesian olasılık ve beklenen kayıp
  * Teknik istatistik detayları (p, CI, z/t, method)

⚙️ Core Capabilities (arka planda çalışır)
- Experiment Setup & Design
  * Hipotez, metric, MDE, trafik dağılımı
  * Otomatik sample size hesaplama
- Data Upload & Integration
  * CSV / XLSX + otomatik kolon algılama
- Data Validation & Quality Checks
  * Sample Ratio Mismatch (SRM)
  * Duplicate, missing, outlier (IQR), veri tipi doğrulama
- Statistical Analysis Engine
  * Metric tipine göre otomatik yöntem (z-test, Welch t-test)
  * p-value, güven aralığı, uplift, effect size
- Guardrail Monitoring
  * Bounce, error, refund, churn, unsubscribe gibi kolonlar otomatik yakalanır
- Bayesian Experimentation
  * Posterior probability, expected loss, %95 HDI
- Advanced Experimentation
  * CUPED (variance reduction)
  * Multiple testing correction (FDR / Bonferroni)
- Segment & Deep Dive
  * Segment bazında uplift + "Confirmed / Exploratory" etiketleri
- Anomaly-Aware Monitoring
  * Zamansal trend grafiği ve sapma izleme
- Decision Engine
  * Ship Winner / Do Not Launch / Continue Test / No Significant Difference / Re-run
- Explainable AI Layer
  * Rule-based yönetici özeti (daima çalışır)
  * LLM tabanlı açıklama (secrets üzerinden yapılandırılır)
- Experiment Knowledge Base
  * Oturum içi deney kaydı ve karşılaştırma

📊 Output & Reporting
- Sonuç kartı (tek bakışta karar + neden)
- Yönetici özeti (rule-based, offline çalışır)
- Teknik özet (p-değer, CI, uplift, örneklem vs.)
- Excel raporu (özet, sonuçlar, guardrail, segment, veri kalitesi sheet'leri)
- JSON raporu (API entegrasyonu)

🗂️ Sürümler / Deployment

Proje iki farklı ortam için paralel olarak paketlenmiştir. Arka plandaki analiz motoru ve UX akışı iki sürümde de aynıdır.

1. Streamlit Cloud (ab-testing/streamlit_app.py)
- Bağımlılıklar: ab-testing/requirements.txt (streamlit>=1.32, pandas, numpy, scipy, plotly, openpyxl, openai>=1.40)
- LLM: Streamlit Cloud → Settings → Secrets altında
  * LLM_API_KEY
  * LLM_BASE_URL (isteğe bağlı; internal endpoint için)
  * LLM_MODEL
- Secrets yoksa yönetici açıklaması rule-based olarak üretilir, LLM butonu bilgilendirme verir.
- Örnek dosya: ab-testing/.streamlit/secrets.toml.example

2. RDP / Offline Windows Terminal (ab-testing/ab-test/app.py)
- Bağımlılıklar: ab-testing/ab-test/requirements.txt
  * streamlit==1.26.0
  * pandas==2.0.3, numpy==1.24.4, scipy==1.11.4
  * plotly==5.17.0, altair==4.2.2, protobuf==4.24.4
  * openpyxl==3.1.2
  * openai==1.2.2
- Çalıştırma: ab-test/run.bat dosyasına çift tıklanır, sanal ortam kurulur ve uygulama http://localhost:8501 üzerinde açılır.
- İnternet erişimi gerekmez; rule-based yönetici özeti hazır gelir.
- İç network LLM endpoint'i varsa ab-test/.streamlit/secrets.toml'da LLM_API_KEY + LLM_BASE_URL + LLM_MODEL tanımlanır; openai==1.2.2 istemcisi iç endpoint'e bağlanır.

🧪 Örnek Test Verileri
ab-testing/test-data altında hazır CSV dosyaları bulunur:
- test_binary_conversion.csv (1.200 satır) — dönüşüm metriği + guardrail + segment
- test_continuous_revenue.csv (1.400 satır) — revenue + pre_revenue (CUPED) + segment + timestamp
- test_multi_variant.csv (1.500 satır) — 3 kol (A/B/C) binary click-through
- test_srm_issue.csv (1.400 satır) — kasıtlı trafik dengesizliği, SRM kontrolünün tetiklenmesini test eder

🔐 Configuration Notes
- secrets.toml.example dosyaları örnek olarak paylaşılmıştır; gerçek değerler asla versiyon kontrolüne eklenmemelidir.
- RDP sürümünde paket kurulumu için iç PyPI aynasına ihtiyaç olabilir.
- Varsayılan değerler (alpha=%5, power=%80) çoğu test için uygundur; Gelişmiş ayarlar expander'ı üzerinden değiştirilebilir.

🚀 Business Impact
Deney süreçlerini standartlaştırır
Yanlış karar riskini azaltır
Teknik olmayan kullanıcıları analiz dokümanı okumaktan kurtarır
Analiz süresini saniyelere indirir
Veri ekibinin destek yükünü hafifletir

💡 Example Use Case
Bir marketing ekibi yeni bir kampanya varyantını test etmek ister:
- Adım 1'de hipotezi yazar, "dönüşüm" seçer ve %2 iyileşme duyarlılığı belirler.
- Adım 2'de CSV yükler; kolonlar otomatik algılanır.
- Adım 3'te "Varyant kazandı, yayına alınabilir" kartını görür, yanında:
  * Dönüşüm %10.5 → %12.1 (+15.3%)
  * Güven: Yeterli (%95)
  * Guardrail: Olumsuz sinyal yok
  * Yönetici özeti tek tık ile kopyalanır.

🧠 Key Differentiators
Üç adımlı sade sihirbaz (13 ekran yerine 3)
Karar odaklı sonuç kartı + istenirse teknik derinlik
Otomatik kolon algılama ve otomatik guardrail yakalama
Rule-based + LLM açıklayıcı
Aynı analiz motoru hem Streamlit Cloud hem offline RDP ortamında
