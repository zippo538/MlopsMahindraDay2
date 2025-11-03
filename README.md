# New York House Price Predction🏠

## 🎯 Deskripsi

Repositori ini berisi implementasi pipeline MLOps yang mencakup tahap-training model, packaging (FastAPI / Streamlit), hingga orchestrasi dengan Docker / docker-compose. Proyek ini dirancang sebagai demonstrasi/pelatihan untuk memperlihatkan bagaimana model machine-learning bisa dibawa dari notebook penelitian ke aplikasi siap produksi.
Folder-folder utamanya meliputi:

notebooks/ → eksplorasi data awal, prototyping model

- `src/` → kode source aplikasi (FastAPI, Streamlit)
- `utils/` → utilitas seperti fungsi pembantu, loader data, dll
- `pages/` → Halaman streamlit
- `static/` → aset statis diperlukan untuk UI atau aplikasi
- `config/` → konfigurasi lingkungan, file YAML/JSON, variabel
- `artifact/` → artefak hasil training (model, metric, logs)
- File root seperti `train.py`, `app.py`, `Home.py`, `docker-compose.yml`, `Dockerfile.*`, `requirements.txt`

## 🚀 Fitur Utama

- Training model melalui train.py
- Aplikasi web front-end (Streamlit) dan/atau REST API (FastAPI)
- Containerisasi dengan Docker (ada Dockerfile.streamlit, Dockerfile.fastapi)
- Orkestrasi multi-container dengan docker-compose.yml
- Struktur modular (src/utils/config) agar mudah dikembangkan sebagai pipeline sesungguhnya

## 🧭 Struktur Proejct
```
/
├── artifact/               # Model dan output training
├── config/                 # Konfigurasi (YAML/JSON/ini)
├── notebooks/              # Notebook eksplorasi data & model
├── src/                    # Kode aplikasi (API, UI, model wrapper)
├── static/                 # Aset untuk UI aplikasi
├── pages /                 # Halaman Streamlit
  ├── 1_👤_Profile.py       # Developer Profile
  ├── 2_📊_Analytics.py     # Data Analitis
  ├── 3_🔮_Predictions.py   # Prediksi Harga
├── utils/                  # Fungsi utilitas (data loader, metrics, logging, etc)
├── .dockerignore
├── .gitignore
├── Dockerfile.streamlit    # Dockerfile untuk aplikasi Streamlit
├── Dockerfile.fastapi      # Dockerfile untuk API FastAPI
├── docker-compose.yml      # Orkestrasi multi-container
├── requirements.txt        # Dependensi Python
├── train.py                # Script utama untuk training model
├── app.py                  # Entry-point REST API (FastAPI)
├── Home.py                 # Entry-point untuk UI (Streamlit) – jika digunakan
└── README.md               # Dokumentasi (Anda ini)
```
## 🛠 Instalasi

- Sebelum dijalankan, pastikan sistem Anda memiliki:
- Python 3.10+ (atau versi yang sesuai di requirements.txt)
- Docker & docker-compose (jika ingin menjalankan container)
- Virtual environment (opsional tapi direkomendasikan)
- Instalasi lokal (tanpa Docker)
- Clone repository

<b>Instalasi lokal (tanpa docker)</b>
1. Clone Directory
```
git clone https://github.com/zippo538/MlopsMahindraDay2.git
cd MlopsMahindraDay2
```
2. Buat virtual environment dan aktifkan
```
python -m venv venv
source venv/bin/activate   # Linux/macOS
# atau `venv\Scripts\activate` untuk Windows
```
3. Instal dependensi
```
pip install -r requirements.txt
```
4. Jalankan training model
```
python train.py
```
5. Jalankan API
```
uvicorn app:app --reload --port 8000
```
6. Jalankan Streamlit
```
streamlit run Home.py
``` 

<b>Instalasi dengan Docker / Docker-Compose </b>
1. FastApi Image
```bash
# build FastApi Image
docker build -t house-fastapi:latest -f Dockerfile.fastapi .

# Run FastApi container
docker run -d -p 8000:8000 --name house-fastapi house-fastapi:latest
```

2. Streamlit Image
```bash
# build Streamlit Image
docker build -t house-streamlit:latest -f Dockerfile.streamlit .

# Run Streamlit container
docker run -d -p 8501:8501 --name house-streamlit house-streamlit:latest
```

3. Akses aplikasi/endpoint sesuai konfigurasi (misalnya `http://localhost:8501` untuk Streamlit atau `http://localhost:8000` untuk API)  

## 🧪 Penggunaan

- Training model: Lihat train.py – latih dan simpan model ke folder artifact/
- API (FastAPI): app.py menerima request (misalnya JSON) dan mengembalikan prediksi berdasarkan model yang sudah dilatih
- UI (Streamlit): Home.py menyediakan antarmuka interaktif untuk pengguna akhir — upload data atau masukkan input manual, lalu dapat prediksi/model insight
- Docker: Setelah deployment dengan Docker, lingkungan ter-isolasi dan siap untuk produksi/testing


## 🧠 Catatan Teknikal

- Logging dan konfigurasi disarankan menggunakan modul logging + config di folder utils/
- Model dapat diperluas dengan pipeline ML (preprocessing, feature engineering, model selection) dan terus-menerus di-monitor
- Untuk produksi, pertimbangkan: versi model (model versioning), logging request, metrik model live, dan CI/CD
- Struktur container berbasis micro-service : API + UI dapat di­-separate jika diperlukan skala lebih besar



## 📚 Referensi / Sumber Belajar

- MLOps: aplikasi praktis dari penelitian ke produksi
- FastAPI: https://fastapi.tiangolo.com
- Streamlit: https://streamlit.io
- Docker & Docker-Compose: dokumentasi resmi
- Struktur proyek rekomendasi untuk ML/AI production

---
📫 For support, email mahindra.irvan538@gmail.com or create an issue in the repository.

Built with ❤️ using Python, FastAPI, and Streamlit
