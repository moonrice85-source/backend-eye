import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

NAMA_INDEX = "my-vector-index"
PINECONE_HOST = "https://my-vector-index-oxdsd3s.svc.aped-4627-b74a.pinecone.io"

# Fungsi pembantu untuk mencari dokumen langsung ke Pinecone menggunakan teks (RAG Cloud)
def cari_dokumen_medis(kata_kunci):
    try:
        # Menembak fitur query teks bawaan dari serverless Pinecone
        url_pinecone = f"{PINECONE_HOST}/query"
        headers_pc = {
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json"
        }
        # Kita minta Pinecone mencari dokumen terdekat berdasarkan teks nama penyakit
        payload_pc = {
            "inputs": kata_kunci,
            "topK": 2,
            "includeMetadata": True
        }
        res_pc = requests.post(url_pinecone, headers=headers_pc, json=payload_pc)
        data_pc = res_pc.json()
        
        konteks_list = []
        if "matches" in data_pc:
            for match in data_pc["matches"]:
                if "metadata" in match and "text" in match["metadata"]:
                    konteks_list.append(match["metadata"]["text"])
        
        return "\n\n".join(konteks_list) if konteks_list else ""
    except:
        return ""

# ==========================================
# 1. ENDPOINT UTAMA UNTUK ANDROID (POST)
# ==========================================
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
        # Ambil dokumen referensi medis dari Pinecone
        konteks_teks = cari_dokumen_medis(nama_penyakit)
        if not konteks_teks:
            konteks_teks = f"Gunakan pengetahuan medis standar mengenai penyakit mata: {nama_penyakit}."

        # Kirim ke Groq AI untuk pembuatan laporan
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
        # Menampilkan detail eror dan isi respons mentah dari API luar jika ada
        error_msg = str(e)
        
        # Coba cek apakah variabel data_groq sempat terbuat untuk melihat isi eror dari Groq
        if 'data_groq' in locals():
            error_msg += f" | Detail Groq: {data_groq}"
        elif 'data_pc' in locals():
            error_msg += f" | Detail Pinecone: {data_pc}"
            
        return jsonify({"status": "error", "message": error_msg}), 500

# ==========================================
# 2. ENDPOINT UJI COBA VIA BROWSER (GET)
# ==========================================
@app.route('/test-katarak', methods=['GET'])
def test_katarak():
    try:
        nama_penyakit = "Katarak"
        
        # Ambil dokumen dari Pinecone
        konteks_teks = cari_dokumen_medis(nama_penyakit)
        if not konteks_teks:
            konteks_teks = "Katarak adalah proses kekeruhan pada lensa mata yang menyebabkan menurunnya visus pasien."

        # Tanya ke Groq AI
        url_groq = "https://api.groq.com/openai/v1/chat/completions"
        headers_groq = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt_teks = f"""Anda adalah seorang AI Asisten Dokter Spesialis Mata yang profesional.
Berikan penjelasan singkat mengenai penyakit Katarak dan rekomendasi tindakan medis awal berdasarkan referensi berikut:\n{konteks_teks}"""

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
            "uji_penyakit": nama_penyakit,
            "respon_llm_groq": hasil_interpretasi
        }), 200
        
    except Exception as e:
        # Menampilkan detail eror dan isi respons mentah dari API luar jika ada
        error_msg = str(e)
        
        # Coba cek apakah variabel data_groq sempat terbuat untuk melihat isi eror dari Groq
        if 'data_groq' in locals():
            error_msg += f" | Detail Groq: {data_groq}"
        elif 'data_pc' in locals():
            error_msg += f" | Detail Pinecone: {data_pc}"
            
        return jsonify({"status": "error", "message": error_msg}), 500
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "active", "message": "Server RAG Vercel (Ultra-Lightweight REST API) Aktif."})
