# Modeling Process

Dokumentasi proses modeling untuk Emotion Classification & Interpretation. Proses ini terdiri dari 3 tahap utama yang dijalankan secara berurutan.

## Overview

1. **Dataset Preparation** - Persiapan dan preprocessing data
2. **Model Training** - Fine-tuning DistilBERT untuk klasifikasi emosi
3. **Interpretation Pipeline** - Menambahkan layer interpretasi menggunakan semantic similarity

---

## 1. Dataset Preparation

### Tujuan
Mempersiapkan dataset EmoTweetID untuk proses training dengan melakukan cleaning dan splitting yang deterministik.

### Proses

1. **Load Dataset**
   - Memuat dataset EmoTweetID dari file CSV
   - Dataset awal: 2,243 sampel dengan 3 kolom (tweet, label, metadata)

2. **Data Cleaning**
   - Menghapus null values dan whitespace
   - Menghapus duplikat exact (tweet + label yang sama)
   - Menghapus konflik label (tweet yang sama dengan label berbeda)
   - Hasil: 2,200 sampel bersih

3. **Label Mapping**
   - Mapping label emosi dari string ke integer:
     - `anger: 0`, `disgust: 1`, `fear: 2`, `joy: 3`, `sadness: 4`, `surprise: 5`
   - Menyimpan mapping ke file JSON untuk konsistensi

4. **Data Splitting**
   - Split stratified untuk menjaga distribusi label:
     - **Train**: 1,540 sampel (70%)
     - **Validation**: 220 sampel (10%)
     - **Test**: 440 sampel (20%)
   - Random state: 42 (untuk reproducibility)

### Output
- File CSV untuk train, validation, dan test
- File `label_mapping.json` untuk mapping label

---

## 2. Model Training

### Tujuan
Melatih model DistilBERT untuk mengklasifikasikan teks Bahasa Indonesia ke dalam 6 kategori emosi.

### Base Model
- **Model**: `cahya/distilbert-base-indonesian`
- **Architecture**: DistilBERT (lightweight BERT)
- **Task**: Sequence Classification (6 kelas emosi)

### Hyperparameters

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `MAX_LENGTH` | 128 | Panjang maksimal token |
| `BATCH_SIZE` | 16 | Ukuran batch untuk training |
| `LEARNING_RATE` | 2.5e-5 | Learning rate untuk fine-tuning |
| `EPOCHS` | 20 | Maksimal epoch training |
| `PATIENCE` | 5 | Early stopping patience |

### Proses Training

1. **Tokenization**
   - Mengubah teks menjadi token IDs menggunakan tokenizer DistilBERT
   - Padding dan truncation ke panjang 128
   - Format: `(batch_size, 128)`

2. **Model Setup**
   - Load pre-trained DistilBERT Indonesian
   - Menambahkan classification head untuk 6 kelas
   - Transfer model ke GPU (Tesla T4)

3. **Training Configuration**
   - **Optimizer**: AdamW dengan weight decay 0.001
   - **Loss Function**: CrossEntropyLoss
   - **Scheduler**: ReduceLROnPlateau (reduce LR jika validation tidak membaik)
   - **Early Stopping**: Stop jika validation F1 tidak membaik selama 5 epoch

4. **Training Loop**
   - Training per epoch dengan validation
   - Metric utama: **Macro F1 Score** (bukan accuracy, karena dataset tidak seimbang)
   - Simpan checkpoint model terbaik berdasarkan validation F1
   - Gradient clipping (max_norm=5.0) untuk stabilitas

5. **Evaluation**
   - Evaluasi pada test set setelah training selesai
   - Metrics: Accuracy, Macro F1, Classification Report, Confusion Matrix

### Hasil Training

- **Best Epoch**: 8
- **Validation Accuracy**: 77.27%
- **Validation F1 (Macro)**: 77.54%
- **Test Accuracy**: 72.05%
- **Test F1 (Macro)**: 72.51%

### Output
- Model terlatih tersimpan di `models/emotion_classifier/`
- File: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`

---

## 3. Interpretation Pipeline

### Tujuan
Menambahkan layer interpretasi yang menjelaskan hasil klasifikasi emosi dalam bahasa natural.

### Pendekatan
Setelah model mengklasifikasikan teks ke dalam salah satu emosi, sistem membandingkan teks input dengan kandidat penjelasan untuk emosi tersebut menggunakan semantic similarity.

### Proses

1. **Kandidat Penjelasan**
   - Setiap emosi memiliki 3 kandidat penjelasan yang telah didefinisikan
   - Contoh untuk "joy":
     - "Teks menunjukkan perasaan senang atau bahagia."
     - "Penulis merasa puas terhadap pengalaman yang dialami."
     - "Isi teks mengandung ekspresi kegembiraan."

2. **Semantic Similarity**
   - Menggunakan embedding model: `intfloat/multilingual-e5-small`
   - Menghitung cosine similarity antara embedding teks input dan embedding kandidat penjelasan
   - Memilih kandidat dengan similarity score tertinggi

3. **Confidence Threshold**
   - Interpretasi hanya diberikan jika confidence klasifikasi >= 0.60
   - Jika confidence rendah, sistem mengembalikan pesan bahwa model tidak cukup yakin

4. **Output**
   - Penjelasan terbaik (best explanation)
   - Similarity score
   - Ranking semua kandidat dengan score masing-masing

### Alur Kerja

```
Input Text
    ↓
[1] Emotion Classification (DistilBERT)
    ↓
Label + Confidence
    ↓
[2] Check Confidence >= 0.60?
    ↓ (Ya)
[3] Semantic Similarity Matching
    ↓
Best Explanation + Ranking
```

### Contoh Output

```python
{
    "label": "joy",
    "confidence": 0.9313,
    "interpretation": "Penulis merasa puas terhadap pengalaman yang dialami.",
    "similarity_score": 0.8396,
    "ranking": [
        {"explanation": "...", "score": 0.8396},
        {"explanation": "...", "score": 0.8297},
        {"explanation": "...", "score": 0.8119}
    ]
}
```

---

## Summary

Proses modeling dimulai dari persiapan dataset yang bersih, kemudian fine-tuning DistilBERT untuk klasifikasi emosi, dan diakhiri dengan penambahan layer interpretasi menggunakan semantic similarity. Hasil akhir adalah sistem yang tidak hanya mengklasifikasikan emosi dengan akurat, tetapi juga memberikan penjelasan yang dapat dipahami manusia.

