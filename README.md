# 🧠 Deteksi Distorsi Kognitif (Bahasa Indonesia)

Aplikasi berbasis **Streamlit + Transformers** untuk mendeteksi pola bahasa yang mengandung **distorsi kognitif** dalam teks berbahasa Indonesia.  
Proyek ini merupakan versi **MVP (Minimum Viable Product)** untuk riset atau eksplorasi awal dalam bidang *Natural Language Processing (NLP)* dan *mental health informatics*.

> ⚠️ **Disclaimer:** Hasil analisis hanya bersifat **informatif** dan **tidak dapat menggantikan** saran atau diagnosis dari profesional kesehatan mental.

---

## 📁 Struktur Folder

Struktur folder proyek ini:
- `Preprocessing.ipynb` — Notebook untuk pembersihan teks (preprocessing)
- `app.py` — Aplikasi utama Streamlit
- `model.ipynb` — Notebook training / eksperimen model
- `preprocessed.csv` — Dataset hasil preprocessing
- `tokenisasi.ipynb` — Notebook tokenisasi dan eksplorasi tokenizer
- `merge_dataset/` — Folder opsional untuk penggabungan dataset
- `train/` — Folder dataset pelatihan
- `validation/` — Folder dataset validasi
- `test/` — Folder dataset pengujian
- `split_data/` — Folder hasil pembagian dataset
- `best_model/` — Folder model terlatih (`config.json`, `tokenizer.json`, `model.safetensors`, dll.)

> Pastikan folder `best_model/` berisi file hasil fine-tuning model seperti:
> - `config.json`
> - `pytorch_model.bin` atau `model.safetensors`
> - `tokenizer.json` / `vocab.txt`  
> Tanpa folder ini, aplikasi **tidak akan bisa dijalankan**.
