# Tugas Besar 2 IF3170 - Prediksi Pembatalan Pemesanan

**Kelompok 31 - Kelompok Dadakan**

Proyek ini memprediksi apakah pemesanan penginapan akan dibatalkan berdasarkan informasi tamu, pemesanan, dan riwayatnya. Target `is_canceled` bernilai `1` untuk pemesanan yang dibatalkan dan `0` untuk yang tidak dibatalkan.

## Anggota

| Nama | NIM |
| --- | --- |
| Farrel Limjaya | 13524046 |
| Raymond Jonathan Dwi Putra Julianto | 13524059 |
| Reynard Anderson Wijaya | 13524111 |
| Fauzan Mohamad Abdul Ghani | 13524113 |
| Hasna Nurul Alimah | 11424061 |

## Struktur proyek

```text
.
|-- data/                       # Data training, testing, dan contoh submission
|-- docs/                       # Dokumen pendukung
|-- output/                     # Model dan submission untuk setiap algoritma
`-- src/
    |-- code/
    |   |-- analysis/           # Eksplorasi dan pemeriksaan data
    |   |-- models/             # Implementasi algoritma dari awal
    |   |-- preprocessing/      # Pembersihan dan transformasi data
    |   `-- workflows/          # Pelatihan, evaluasi, dan submission
    `-- notebooks/
        |-- IF3170_G31_KelompokDadakan.ipynb
        `-- Template Tubes.ipynb
```

## Data

Folder `data/` berisi tiga file yang digunakan dalam proses pelatihan dan submission:

- `train.csv`: fitur pemesanan dan label `is_canceled`.
- `test.csv`: fitur pemesanan tanpa label.
- `sample_submission.csv`: contoh format submission.

Kolom `id` dibaca sebagai teks dan tidak digunakan sebagai fitur. Hasil submission berisi kolom `id` dan `is_canceled`, dengan satu baris untuk setiap pemesanan pada data test.

## Persiapan dan penggunaan

Proyek menggunakan Python dengan NumPy, pandas, SciPy, Matplotlib, scikit-learn, IPython, dan Jupyter. Instalasi dependensi dapat dilakukan melalui perintah berikut:

```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn ipython ipykernel jupyter
```

Notebook utama adalah `src/notebooks/IF3170_G31_KelompokDadakan.ipynb`. Jupyter dapat dijalankan dari folder proyek dengan perintah berikut:

```bash
jupyter notebook
```

Sel notebook dijalankan secara berurutan. Lokasi proyek dikenali berdasarkan keberadaan folder `data/` dan `src/`. Pelatihan seluruh model dan cross-validation, terutama SVM, dapat memerlukan waktu. Sel penyimpanan model dan submission akan menulis ulang file keluaran dengan nama yang sama di dalam `output/`.

## Alur pengerjaan

Data diperiksa dan dibagi secara stratifikasi menjadi data training dan validasi sebelum preprocessing dipelajari. Tahap persiapan meliputi pemeriksaan nilai kosong, duplikasi, nilai ekstrem, imputasi, standardisasi, dan encoding. Beberapa model menggunakan transformasi tambahan yang dipelajari dari bagian training.

Lima algoritma diimplementasikan dari awal dan dibandingkan dengan scikit-learn:

- K-Nearest Neighbors (KNN)
- Decision Tree Learning (DTL) menggunakan ID3
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes

Kinerja dievaluasi menggunakan **Macro F1**, akurasi, F1 setiap kelas, dan confusion matrix. Macro F1 menjadi metrik utama kompetisi dan memberikan bobot yang sama kepada kedua kelas. Notebook juga memuat stratified 5-fold cross-validation. Model final dilatih kembali menggunakan seluruh data training sebelum membuat submission.

## Keluaran

Hasil setiap model disimpan dalam foldernya masing-masing di `output/`. Contohnya, folder `output/SVM/` berisi model, preprocessor, dan `submission_svm.csv`. File pickle digunakan untuk menyimpan model dan preprocessor agar dapat dimuat kembali dari lingkungan Python yang sesuai.
