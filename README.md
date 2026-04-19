🧪 A/B Testing Agent

📌 Overview
A/B Testing Agent, kullanıcıların deney tasarımı, veri doğrulama, istatistiksel analiz ve karar üretim süreçlerini tek bir platform üzerinden yönetmesini sağlayan uçtan uca bir deney yönetim aracıdır.
Bu agent, teknik bilgisi sınırlı kullanıcıların dahi güvenilir A/B testleri kurgulamasını, analiz etmesini ve sonuçları iş kararlarına dönüştürmesini mümkün kılar.
Agent; deney kurulumundan veri validasyonuna, istatistiksel analizden aksiyon önerilerine kadar tüm süreci otomatikleştirir ve standartlaştırır.

🎯 Project Purpose
Kurum içinde deney kültürünü yaygınlaştırmak
Business ekiplerin teknik destek almadan A/B test yapabilmesini sağlamak
Yanlış test kurgularını ve hatalı yorumlamaları engellemek
Karar alma süreçlerini hızlandırmak ve veri odaklı hale getirmek

👥 Target Users
Marketing & Growth Teams
Product Managers
Digital Analytics Teams
CRM & Campaign Teams
Data Analysts

⚙️ Core Capabilities
1. Experiment Setup & Design
Hipotez tanımlama
Kontrol ve varyant gruplarının belirlenmesi
Primary metric ve guardrail metric seçimi
Trafik dağılımı tanımı (A/B/C vs.)
Confidence level ve power ayarları
Minimum Detectable Effect (MDE) belirleme
Otomatik sample size hesaplama

2. Data Upload & Integration
CSV / XLSX veri yükleme
Kolon eşleme (variant, metric, timestamp vb.)
Veri önizleme
Multi-series ve multi-experiment desteği
Batch experiment analizi

3. Data Validation & Quality Checks
Agent, test sonuçlarının güvenilirliğini sağlamak için otomatik veri kontrolleri yapar:
Sample Ratio Mismatch (SRM) detection
Duplicate kayıt kontrolü
Missing data analizi
Randomization kontrolü
Outlier detection (IQR tabanlı)
Metric sanity checks
Data type validation

4. Statistical Analysis Engine
Agent, metrik tipine göre doğru istatistiksel yöntemi otomatik seçer:
Supported Metrics:
Binary metrics (conversion, click)
Continuous metrics (revenue, duration)
Count metrics
Rate metrics
Retention metrics
Supported Methods:
z-test / chi-square
t-test / Welch’s test
Non-parametric tests (Mann-Whitney U)
Bayesian A/B testing
Output:
p-value
confidence interval
uplift (% ve absolute)
effect size (Cohen's d)

5. Guardrail Monitoring
Primary metric dışında yan etkileri izlemek için:
churn rate
refund rate
bounce rate
error rate
unsubscribe rate
revenue impact
Agent, şu tarz uyarılar üretir:
Primary metric improved, but guardrail metrics show negative impact.

6. Sequential Testing & Smart Monitoring
Peeking-safe sequential analysis (O’Brien-Fleming tarzı alpha spending)
Test devam ederken güvenli yorumlama
Automatic early stopping önerileri
Dynamic confidence tracking

7. Bayesian Experimentation
Posterior probability hesaplama
Probability of B being better than A
Expected loss (risk) hesabı
Risk-based decision support
Uncertainty-aware recommendations

8. Advanced Experimentation Features
Multi-Armed Bandit Support (roadmap)
Trafik dinamik olarak en iyi varyanta yönlendirilir
Exploration vs exploitation dengesi
CUPED (Variance Reduction)
Daha hızlı ve daha hassas sonuçlar
Daha düşük sample size ihtiyacı
Multiple Testing Correction
False positive riskini azaltma
Bonferroni / FDR (Benjamini-Hochberg) düzeltmeleri

9. Segment & Deep Dive Analysis
Device bazlı analiz
Country / region kırılımı
User segment analizi
New vs returning users
Channel bazlı performans
Agent, segment sonuçlarını:
Exploratory insight
Confirmed result
olarak ayırır.

10. Anomaly-Aware Experiment Monitoring
Test süresince anomali tespiti (z-skoru tabanlı)
Ani metrik değişimlerinin yakalanması
Data drift ve davranış değişikliklerinin analizi

11. What-If Simulation
“Conversion %3 artarsa ne olur?”
“Traffic %20 artarsa sonuç nasıl değişir?”
Senaryo bazlı tahminleme

12. Decision Engine
Agent, analiz sonuçlarını aksiyona dönüştürür:
✅ Ship Winner
⏳ Continue Test
⚖️ No Significant Difference
❌ Do Not Launch
🔁 Re-run Experiment
Karar önerileri:
istatistiksel güven
iş etkisi
veri yeterliliği
baz alınarak üretilir.

13. Explainable AI Layer (LLM Support)
Sonuçların doğal dilde açıklanması
Executive summary üretimi
Teknik detayların sadeleştirilmesi
“Bu test neden böyle sonuç verdi?” açıklamaları
İki katman:
- Rule-based offline açıklayıcı (daima çalışır)
- LLM tabanlı açıklayıcı (secrets üzerinden yapılandırılır)

14. Experiment Knowledge Base
Geçmiş testlerin oturum içinde saklanması
Benzer deneylerin önerilmesi
Öğrenen sistem yapısı
Test başarı / başarısızlık pattern analizi

📊 Output & Reporting
1. Executive Summary
Kazanan varyant
Güven seviyesi
İş etkisi
Önerilen aksiyon
2. Analytical Report
p-value
confidence interval
uplift
segment sonuçları
3. Technical Validation Log
SRM sonucu
kullanılan test yöntemi
veri temizleme adımları
varsayım kontrolleri

📤 Export Options
Excel (detaylı sonuçlar, çok sekmeli rapor)
JSON (API entegrasyonu)
Dashboard çıktıları (uygulama içi canlı görünüm)
PDF export (roadmap)

🧩 System Architecture 
Experiment Setup
Data Ingestion
Validation Engine
Statistical Engine
Decision Engine
Reporting & Export

🚀 Business Impact
Deney süreçlerini standartlaştırır
Yanlış karar riskini azaltır
Analiz süresini ciddi ölçüde kısaltır
Teknik bağımlılığı azaltır
Daha hızlı ve güvenilir ürün geliştirme sağlar

💡 Example Use Case
Bir marketing ekibi yeni bir kampanya varyantını test etmek ister:
Kullanıcı hipotezi tanımlar
Veri yüklenir
Agent otomatik analiz yapar
Sonuçları yorumlar
“B varyantı %6 uplift ile kazandı, rollout önerilir” çıktısını verir

🧠 Key Differentiators
End-to-end experimentation flow
Built-in validation (SRM, data quality)
Advanced statistical methods (Bayesian, CUPED)
Decision-oriented outputs
Non-technical user friendly design
Explainable results (rule-based + LLM)

🗂️ Sürümler / Deployment

Proje iki farklı ortam için paralel olarak paketlenmiştir:

1. Streamlit Cloud (streamlit_app.py)
- Dosya: ab-testing/streamlit_app.py
- Bağımlılıklar: ab-testing/requirements.txt (streamlit>=1.32, pandas, numpy, scipy, plotly, openpyxl, openai>=1.40)
- LLM entegrasyonu: Streamlit Cloud "Settings → Secrets" altında aşağıdaki anahtarlar tanımlanır:
  - LLM_API_KEY
  - LLM_BASE_URL
  - LLM_MODEL
- Secrets tanımlı değilse uygulama çalışmaya devam eder, yalnızca LLM açıklama adımı devre dışı kalır.
- Örnek için ab-testing/.streamlit/secrets.toml.example bakılabilir.

2. RDP / Offline Windows Terminal (ab-test/app.py)
- Dosya: ab-testing/ab-test/app.py
- Bağımlılıklar: ab-testing/ab-test/requirements.txt
  - streamlit==1.26.0
  - pandas==2.0.3, numpy==1.24.4, scipy==1.11.4
  - plotly==5.17.0, altair==4.2.2, protobuf==4.24.4
  - openpyxl==3.1.2
  - openai==1.2.2
- Kurulum / çalıştırma: ab-test klasöründeki run.bat dosyasına çift tıklanır. Sanal ortam
  oluşturulur, requirements.txt kurulumu yapılır ve uygulama http://localhost:8501 üzerinde
  başlatılır.
- İnternet bağlantısı gerektirmez. Varsayılan olarak rule-based offline açıklama çalışır.
- Kurum içi bir LLM endpoint'i mevcutsa ab-test/.streamlit/secrets.toml dosyasına
  LLM_API_KEY, LLM_BASE_URL ve LLM_MODEL yazıldığında openai 1.2.2 istemcisi ile
  iç network üzerinden açıklama üretilebilir.

🔐 Configuration Notes
- secrets.toml.example dosyaları örnek olarak paylaşılmıştır; gerçek değerler asla
  versiyon kontrolüne eklenmemelidir.
- RDP sürümünde paket kurulumu için iç PyPI aynasına ihtiyaç olabilir; organizasyonun
  sunduğu indeksi kullanmak gerekir.
- Sample size ve sequential testing hesaplamaları alpha / power / MDE ayarlarına duyarlıdır;
  deney tasarımı ekranında bu parametreler girilmelidir.
