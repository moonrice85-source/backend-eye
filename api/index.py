import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
# Menggunakan alternatif embedding berbasis API agar Vercel tidak download model berat
from langchain_community.embeddings import HuggingFaceInferenceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. AMBIL PATH WEIGHTS YOLO LOKAL (Gunakan Versi Nano)
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_YOLO_WEIGHTS = os.path.join(BASE_DIR, "weights", "best.pt")

# Memuat YOLO secara malas (lazy loading) di dalam fungsi nanti agar tidak membebani start-up Vercel
model_yolo = None

def get_yolo_model():
    global model_yolo
    if model_yolo is None:
        model_yolo = YOLO(PATH_YOLO_WEIGHTS)
    return model_yolo

# ==========================================
# 2. INISIALISASI EMBEDDING & PINECONE (API BASED)
# ==========================================
# Kita gunakan HuggingFaceInferenceEmbeddings agar proses embedding dilakukan di cloud Hugging Face (Gratis & Ringan)
model_embedding = HuggingFaceInferenceEmbeddings(
    api_key=os.environ.get("GROQ_API_KEY"), # Menggunakan key yang ada untuk trigger library
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

NAMA_INDEX_PINECONE = "my-vector-index" 

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
    path_sementara = "/tmp/upload_android.jpg"
    file_gambar.save(path_sementara)
    
    try:
        # Panggil model YOLO saat ada request masuk saja
        yolo = get_yolo_model()
        results = yolo.predict(source=path_sementara, conf=0.25, verbose=False)
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
        
        # Koneksi ke Pinecone secara realtime saat dibutuhkan
        db_vektor = PineconeVectorStore.from_existing_index(
            index_name=NAMA_INDEX_PINECONE,
            embedding=model_embedding
        )
        
        # TAHAP 2: Retrieval Dokumen Medis dari Pinecone Cloud
        dokumen_cocok = db_vektor.similarity_search(nama_penyakit, k=2) # Dikurangi ke k=2 agar token tidak kepenuhan
        konteks_teks = "\n\n".join([doc.page_content for doc in dokumen_cocok])
        
        # TAHAP 3: Generation Laporan Medis via Groq AI
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.2)
        chain = prompt | llm
        respons = chain.invoke({
            "hasil_yolo": nama_penyakit, 
            "confidence": nilai_confidence, 
            "dokumen_konteks": konteks_teks
        })
        
        if os.path.exists(path_sementara):
            os.remove(path_sementara)
            
        return jsonify({
            "status": "success",
            "penyakit": nama_penyakit,
            "confidence": f"{nilai_confidence:.2f}%",
            "interpretasi": respons.content
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "active", "message": "Server API EyeBot RAG-YOLO siap melayani Android."})
