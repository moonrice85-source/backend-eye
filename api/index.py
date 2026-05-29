import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.embeddings import HuggingFaceInferenceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# Inisialisasi Embedding API (Sangat Ringan)
model_embedding = HuggingFaceInferenceEmbeddings(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

NAMA_INDEX_PINECONE = "my-vector-index" 

# Template Prompt Medis RAG
template_prompt = """
Anda adalah seorang AI Asisten Dokter Spesialis Mata yang sangat profesional.
Tugas Anda adalah memberikan interpretasi klinis dan langkah penanganan medis berdasarkan hasil deteksi gambar serta dokumen referensi medis terpercaya yang disediakan di bawah ini. Jawablah menggunakan Bahasa Indonesia yang formal dan terstruktur.

[HASIL DETEKSI SISTEM YOLOv8]
Penyakit Terdeteksi: {hasil_yolo}
Tingkat Keyakinan (Confidence): {confidence}

[DOKUMEN REFERENSI MEDIS]
{dokumen_konteks}

[PERINTAH]
Berikan penjelasan mendalam mengenai penyakit tersebut, tanda-tandanya pada citra fundus berdasarkan referensi, dan berikan rekomendasi tindakan medis awal yang harus dilakukan pasien.
"""
prompt = PromptTemplate(input_variables=["hasil_yolo", "confidence", "dokumen_konteks"], template=template_prompt)

# ENDPOINT BARU: Menerima teks nama penyakit dari Android atau Colab
@app.route('/generate-interpretation', methods=['POST'])
def generate_interpretation():
    data = request.json
    if not data or 'penyakit' not in data or 'confidence' not in data:
        return jsonify({"status": "error", "message": "Format request salah. Butuh 'penyakit' dan 'confidence'"}), 400
        
    nama_penyakit = data['penyakit']
    nilai_confidence = data['confidence']
    
    try:
        # Jika mata normal, tidak perlu cari dokumen di Pinecone
        if nama_penyakit.lower() == "normal":
            return jsonify({
                "status": "success",
                "interpretasi": "Hasil analisis citra fundus menunjukkan kondisi mata normal. Tidak ditemukan indikasi kelainan struktural."
            }), 200

        # Koneksi ke Pinecone Cloud
        db_vektor = PineconeVectorStore.from_existing_index(
            index_name=NAMA_INDEX_PINECONE,
            embedding=model_embedding
        )
        
        # Ambil dokumen relevan
        dokumen_cocok = db_vektor.similarity_search(nama_penyakit, k=2)
        konteks_teks = "\n\n".join([doc.page_content for doc in dokumen_cocok])
        
        # Generate laporan via Groq
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.2)
        chain = prompt | llm
        respons = chain.invoke({
            "hasil_yolo": nama_penyakit, 
            "confidence": nilai_confidence, 
            "dokumen_konteks": konteks_teks
        })
        
        return jsonify({
            "status": "success",
            "interpretasi": respons.content
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "active", "message": "Server RAG Vercel Aktif."})
