import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)  # Mengizinkan koneksi aman dari aplikasi Android Studio

# ==========================================
# 1. AMBIL PATH WEIGHTS YOLO LOKAL
# ==========================================
# Menggunakan path relatif agar Vercel bisa membaca file di dalam folder proyek
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_YOLO_WEIGHTS = os.path.join(BASE_DIR, "weights", "best.pt")

print("Memuat model YOLOv8 dari:", PATH_YOLO_WEIGHTS)
model_yolo = YOLO(PATH_YOLO_WEIGHTS)

# ==========================================
# 2. INISIALISASI EMBEDDING & PINECONE (RAG)
# ==========================================
# Catatan: API Key untuk Groq dan Pinecone tidak ditulis keras (hardcode) di sini,
# melainkan akan kita masukkan dengan aman lewat dashboard Environment Variables Vercel.

model_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Nama index harus sama persis dengan yang kamu buat di website Pinecone kemarin
NAMA_INDEX_PINECONE = "pcsk_5W82ip_AVt2YYgWDYTEPPgLnEeKMbpSJNwaDzE45rj2wArVtq4rQHbF98KRWYuH7fo4Dn7" 

# Inisialisasi koneksi database vektor cloud
db_vektor = PineconeVectorStore.from_existing_index(
    index_name=my-vector-index,
    embedding=model_embedding
)

# Inisialisasi LLM Groq AI
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.2)

# Template Prompt Medis RAG
template_prompt = """
Anda adalah seorang AI Asisten Dokter Spesialis Mata yang sangat profesional.
Tugas Anda adalah memberikan interpretasi klinis dan langkah penanganan medis berdasarkan hasil deteksi gambar (YOLOv8) serta dokumen referensi medis terpercaya yang disediakan di bawah ini. Jawablah menggunakan Bahasa Indonesia yang formal dan terstruktur.

[HASIL DETEKSI SISTEM YOLOv8]
Penyakit Terdeteksi: {hasil_yolo}
Tingkat Keyakinan (Confidence): {confidence:.2f}%

[DOKUMEN REFERENSI MEDIS]
{dokumen_konteks}

[PERINTAH]
Berikan penjelasan mendalam mengenai penyakit tersebut, tanda-tandanya pada citra fundus berdasarkan referensi, dan berikan rekomendasi tindakan medis awal yang harus dilakukan pasien.
"""
prompt = PromptTemplate(input_variables=["hasil_yolo", "confidence", "dokumen_konteks"], template=template_prompt)

# ==========================================
# 3. ENDPOINT API UNTUK DIEKSEKUSI ANDROID
# ==========================================
@app.route('/predict', methods=['POST'])
def predict_fundus():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "Format request salah, pastikan kunci bernama 'image'"}), 400
        
    file_gambar = request.files['image']
    path_sementara = "/tmp/upload_android.jpg"  # Vercel hanya mengizinkan penulisan file di folder /tmp
    file_gambar.save(path_sementara)
    
    try:
        # TAHAP 1: Deteksi Citra Mata dengan YOLOv8
        results = model_yolo.predict(source=path_sementara, conf=0.25, verbose=False)
        result = results[0]
        
        if len(result.boxes) == 0:
            return jsonify({
                "status": "success",
                "penyakit": "Normal",
                "confidence": "0.00%",
                "interpretasi": "Hasil analisis citra fundus menunjukkan kondisi mata normal. Tidak ditemukan indikasi kelainan struktural."
            }), 200
            
        box_tertinggi = result.boxes[0]
        id_kelas = int(box_tertinggi.cls[0])
        nama_penyakit = result.names[id_kelas]
        nilai_confidence = float(box_tertinggi.conf[0]) * 100
        
        # TAHAP 2: Retrieval Dokumen Medis dari Pinecone Cloud
        dokumen_cocok = db_vektor.similarity_search(nama_penyakit, k=3)
        konteks_teks = "\n\n".join([doc.page_content for doc in dokumen_cocok])
        
        # TAHAP 3: Generation Laporan Medis via Groq AI
        chain = prompt | llm
        respons = chain.invoke({
            "hasil_yolo": nama_penyakit, 
            "confidence": nilai_confidence, 
            "dokumen_konteks": konteks_teks
        })
        
        # Hapus berkas temporary
        if os.path.exists(path_sementara):
            os.remove(path_sementara)
            
        # Kembalikan data dalam bentuk JSON murni ke Android Studio
        return jsonify({
            "status": "success",
            "penyakit": nama_penyakit,
            "confidence": f"{nilai_confidence:.2f}%",
            "interpretasi": respons.content
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Endpoint tambahan untuk memastikan status server di Vercel aktif
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "active", "message": "Server API EyeBot RAG-YOLO siap melayani Android."})
