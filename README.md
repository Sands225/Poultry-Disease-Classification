# Deteksi Penyakit Unggas Berbasis Citra Feses Menggunakan YOLOv11

Proyek ini merupakan implementasi sistem deteksi objek berbasis *Deep Learning* untuk mengidentifikasi kondisi kesehatan unggas melalui citra feses. Model dikembangkan dengan melakukan *fine-tuning* pada arsitektur **YOLOv11 Nano (`yolo11n`)** dan diintegrasikan ke dalam layanan antarmuka pemrograman aplikasi (API) menggunakan **FastAPI**.

## Latar Belakang dan Kelas Target

Deteksi dini penyakit pada unggas melalui pengamatan visual feses membantu peternak dalam memantau kesehatan ternak secara cepat. Model dalam proyek ini dilatih untuk mengenali dan melokalisasi empat kelas kondisi feses unggas:

| ID Kelas | Label | Keterangan |
| :---: | :--- | :--- |
| 0 | `cocci` | Coccidiosis |
| 1 | `healthy` | Kondisi sehat (normal) |
| 2 | `ncd` | Newcastle Disease |
| 3 | `salmo` | Salmonella |

---

## Struktur Direktori

```text
.
├── api/
│   ├── main.py                      # Implementasi REST API menggunakan FastAPI
│   └── YOLOv11n_Class Weight.pt     # Bobot model YOLOv11n hasil pelatihan
├── .gitignore
└── README.md
```

---

## Alur Kerja Sistem

Secara garis besar, pengembangan sistem ini terbagi menjadi dua tahap utama, yaitu pelatihan model (*training pipeline*) dan penyajian model melalui API (*inference pipeline*).

### 1. Tahap Pelatihan Model
1. **Pengumpulan dan Penyeragaman Dataset**
   Data citra dan anotasi dikumpulkan dari beberapa sumber, meliputi dataset publik (Kaggle/Zenodo dan Roboflow) serta data anotasi segmentasi yang disimpan di Google Drive. Seluruh label diseragamkan ke dalam empat kelas target (`cocci`, `healthy`, `ncd`, `salmo`).
2. **Audit dan Pembersihan Data**
   Sebelum pelatihan, dilakukan pemeriksaan otomatis terhadap integritas berkas untuk mendeteksi citra yang rusak (*corrupt*), label yang kosong atau hilang, koordinat *bounding box* di luar batas normal, serta citra duplikat menggunakan perbandingan *hash* MD5. Berkas yang bermasalah dan duplikat dibersihkan agar tidak memengaruhi kualitas pelatihan.
3. **Pembagian Dataset (*Data Splitting*)**
   Dataset bersih dibagi menjadi tiga himpunan dengan proporsi **70% data latih (*train*)**, **15% data validasi (*valid*)**, dan **15% data uji (*test*)**.
4. **Penanganan Ketidakseimbangan Kelas (*Class Imbalance*)**
   Distribusi data antar kelas yang tidak seimbang—khususnya pada kelas `ncd` yang jumlahnya lebih sedikit—ditangani melalui eksperimen pembobotan kelas (*class weight* bawaan Ultralytics `cls_pw`) yang menghasilkan bobot akhir `YOLOv11n_Class Weight.pt`, serta pendekatan *Random Oversampling* (ROS) pada himpunan latih sebagai pembanding.
5. **Pelatihan dan Evaluasi Model**
   Model `yolo11n.pt` dilatih ulang (*fine-tuning*) menggunakan pustaka Ultralytics. Performa model dievaluasi pada data validasi dan data uji menggunakan metrik standar deteksi objek, yaitu *Precision*, *Recall*, *mAP50*, dan *mAP50-95*.

### 2. Tahap Inferensi melalui API
1. Klien mengirimkan berkas citra feses melalui permintaan HTTP `POST` ke endpoint `/predict`.
2. Server membaca aliran *byte* gambar dan melakukan pra-pemrosesan dengan mengonversinya ke format citra RGB menggunakan Pillow (`PIL`).
3. Model `YOLOv11n_Class Weight.pt` menjalankan inferensi dengan ukuran masukan `imgsz=640` dan ambang batas kepercayaan (*confidence threshold*) default `0.4`.
4. Hasil deteksi berupa nama kelas, nilai kepercayaan (*confidence score*), dan koordinat *bounding box* (`xyxy`) dikembalikan kepada klien dalam format JSON.

---

## Cara Menjalankan API

### Prasyarat
Pastikan **Python 3.10** atau versi yang lebih baru telah terpasang pada sistem Anda, kemudian pasang pustaka yang dibutuhkan:

```bash
pip install fastapi uvicorn ultralytics pillow python-multipart
```

Pastikan pula berkas bobot model `YOLOv11n_Class Weight.pt` berada di dalam direktori `api/` sejajar dengan `main.py`.

### Menjalankan Server Lokal
Jalankan perintah berikut dari direktori utama (*root*) proyek:

```bash
uvicorn api.main:app --reload
```

Atau apabila dijalankan dari dalam direktori `api/`:

```bash
uvicorn main:app --reload
```

Setelah server berjalan, layanan dapat diakses melalui alamat `http://127.0.0.1:8000`. Dokumentasi interaktif (Swagger UI) tersedia pada `http://127.0.0.1:8000/docs` untuk menguji endpoint secara langsung melalui peramban.

---

## Dokumentasi Endpoint API

### 1. Cek Status Layanan
- **Endpoint**: `GET /`
- **Deskripsi**: Memastikan server API dalam kondisi aktif.
- **Contoh Respons**:
  ```json
  {
    "status": "online"
  }
  ```

### 2. Prediksi Penyakit Unggas
- **Endpoint**: `POST /predict`
- **Parameter**:
  - `file` (*form-data*, wajib): Berkas gambar feses unggas (JPG/PNG).
  - `conf` (*query*, opsional, default `0.4`): Ambang batas minimum *confidence score* deteksi.
- **Contoh Pengujian dengan `curl`**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/predict?conf=0.4" \
       -F "file=@contoh_feses.jpg"
  ```
- **Contoh Respons**:
  ```json
  {
    "detections": [
      {
        "class_name": "cocci",
        "confidence": 0.8921,
        "bbox": [112.45, 84.12, 340.88, 295.67]
      }
    ]
  }
  ```