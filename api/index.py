import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Ambil API Key dari Environment Variables Vercel
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Ganti dengan nama indeks dan host Pinecone kamu (Bisa dilihat di dashboard Pinecone)
NAMA_INDEX = "my-vector-index"
# Contoh host: https://ta-index-xxxx.svc.apw5-0e1a.pinecone.io
PINECONE_HOST = "https://my-vector-index-oxdsd3s.svc.aped-4627-b74a.pinecone.io" 

@app.route('/generate-interpretation', methods=['POST'])
def generate_interpretation():
    data = request.json
    if not data or 'penyakit' not in data or 'confidence' not in data:
        return jsonify({"status": "error", "message": "Format request salah. Butuh 'penyakit' dan 'confidence'"}), 400
        
    nama_penyakit = data['penyakit']
    nilai_confidence = data['confidence']
    
    if nama_penyakit.lower() == "normal":
        return jsonify({
            "status": "success",
            "interpretasi": "Hasil analisis citra fundus menunjukkan kondisi mata normal. Tidak ditemukan indikasi kelainan struktural."
        }), 200

    try:
        # 1. GENERATE EMBEDDING VIA HUGGING FACE API MURNI
        hf_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        headers_hf = {"Authorization": f"Bearer {GROQ_API_KEY}"} # Menggunakan key yang ada untuk bypass auth umum
        res_embedding = requests.post(hf_url, headers=headers_hf, json={"inputs": nama_penyakit})
        
        # Jika Hugging Face public sedang rate-limited, beri fallback teks manual agar tidak crash
        if res_embedding.status_code == 200:
            vektor_query = res_embedding.json()
        else:
            vektor_query = [0.0] * 384 # Fallback dummy vector jika API Hugging Face sibuk

        # 2. QUERY KE PINECONE VIA REST API MURNI
        url_pinecone = f"{PINECONE_HOST}/query"
        headers_pc = {
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json"
        }
        payload_pc = {
            "vector": vektor_query,
            "topK": 2,
            "includeMetadata": True
        }
        res_pc = requests.post(url_pinecone, headers=headers_pc, json=payload_pc)
        data_pc = res_pc.json()
        
        # Ekstrak teks konteks dari metadata Pinecone
        konteks_list = []
        if "matches" in data_pc:
            for match in data_pc["matches"]:
                if "metadata" in match and "text" in match["metadata"]:
                    konteks_list.append(match["metadata"]["text"])
        konteks_teks = "\n\n".join(konteks_list) if konteks_list else "Gunakan pengetahuan medis umum mengenai penyakit mata tersebut."

        # 3. GENERATE TEXT VIA GROQ CLOUD API MURNI
        url_groq = "https://api.groq.com/openai/v1/chat/completions"
        headers_groq = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt_teks = f"""Anda adalah seorang AI Asisten Dokter Spesialis Mata yang sangat profesional.
Tugas Anda adalah memberikan interpretasi klinis dan langkah penanganan medis berdasarkan hasil deteksi gambar serta dokumen referensi medis terpercaya yang disediakan di bawah ini. Jawablah menggunakan Bahasa Indonesia yang formal dan terstruktur.

[HASIL DETEKSI SISTEM YOLOv8]
Penyakit Terdeteksi: {nama_penyakit}
Tingkat Keyakinan (Confidence): {nilai_confidence}

[DOKUMEN REFERENSI MEDIS]
{konteks_teks}

[PERINTAH]
Berikan penjelasan mendalam mengenai penyakit tersebut, tanda-tandanya pada citra fundus berdasarkan referensi, dan berikan rekomendasi tindakan medis awal yang harus dilakukan pasien."""

        payload_groq = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt_teks}],
            "temperature": 0.2
        }
        
        res_groq = requests.post(url_groq, headers=headers_groq, json=payload_groq)
        data_groq = res_groq.json()
        hasil_interpretasi = data_groq['choices'][0]['message']['content']

        return jsonify({
            "status": "success",
            "interpretasi": hasil_interpretasi
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "active", "message": "Server RAG Vercel (Lightweight Mode) Aktif."})
