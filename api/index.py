import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ==========================================
# CONFIGURATION & CLOUD CREDENTIALS
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST = os.environ.get("PINECONE_HOST")

# =========================================================================
# MASUKKAN FILE ID GOOGLE DRIVE DARI FILE best.pt UNTUK DOWNLOAD OTOMATIS
# =========================================================================
DRIVE_FILE_ID = "1qkCrQDRoRAPHkmW6i67nujL_JLDHwC33"
MODEL_PATH = "/tmp/best.pt"

# Daftar penyakit mata yang VALID sesuai fokus penelitian
VALID_DISEASES = {"normal", "katarak", "konjungtivitis", "uveitis", "pterygium", "hordeolum"}

def download_model_from_drive():
    """Fungsi mendownload file best.pt dari Drive jika belum ada di folder /tmp Vercel"""
    if not os.path.exists(MODEL_PATH):
        print("Mengunduh model best.pt asli dari Google Drive...")
        url = f"https://docs.google.com/uc?export=download&id={DRIVE_FILE_ID}"
        response = requests.get(url, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Unduhan file best.pt selesai!")

def normalize_label(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())

# ==========================================
# LOGIKA INTI: INTEGRASI RAG MATA
# ==========================================
def process_eye_rag(nama_penyakit: str, nilai_confidence: str, lang: str, top_k: int = 3):
    clean_disease = normalize_label(nama_penyakit)
    
    if clean_disease not in VALID_DISEASES:
        return jsonify({"status": "error", "message": f"Penyakit '{nama_penyakit}' di luar cakupan penelitian."}), 400

    if clean_disease == "normal":
        msg_normal = ("Based on the external eye image analysis, your eye condition appears normal." if lang.lower() == "en" else "Hasil analisis citra eksternal mata menunjukkan kondisi mata normal.")
        return jsonify({
            "status": "success", "disease_name": "Normal", "confidence": nilai_confidence,
            "retrieved_context": "N/A", "page": None, "paragraph": None, "retrieved_contexts": [], "interpretation": msg_normal
        }), 200

    try:
        url_pinecone = f"{PINECONE_HOST}/query"
        headers_pc = {"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"}
        query_internal = f"Clinical signs, symptoms, visual presentation, and medical treatment for {clean_disease}."
        
        payload_pc = {"inputs": {"text": query_internal}, "topK": top_k, "includeMetadata": True, "filter": {"disease": {"$eq": clean_disease}}}
        res_pc = requests.post(url_pinecone, headers=headers_pc, json=payload_pc)
        data_pc = res_pc.json()
        
        retrieved_contexts = []
        rank_counter = 1
        if "matches" in data_pc:
            for match in data_pc["matches"]:
                metadata = match.get("metadata", {})
                retrieved_contexts.append({
                    "rank": rank_counter, "context": metadata.get("text", "Deskripsi tidak tersedia."),
                    "page": metadata.get("page", 1), "paragraph": metadata.get("paragraph", 1),
                    "chunk_index": metadata.get("chunk_index", rank_counter), "similarity_score": float(match.get("score", 0.0))
                })
                rank_counter += 1

        if not retrieved_contexts:
            retrieved_contexts.append({"rank": 1, "context": f"Tata laksana umum penyakit {clean_disease}.", "page": 1, "paragraph": 1, "chunk_index": 1, "similarity_score": 0.5})

        combined_context = "\n\n".join([f"[Referensi {i['rank']} | Halaman {i['page']}]\n{i['context']}" for i in retrieved_contexts])

        # PROMPT INTERPRETASI MEDIS VIA GROQ
        url_groq = "https://api.groq.com/openai/v1/chat/completions"
        headers_groq = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        
        system_msg = "Anda adalah AI Asisten Dokter Spesialis Mata. Berikan interpretasi klinis 2 paragraf berdasarkan hasil deteksi YOLOv8 dan referensi buku RAG."
        prompt = f"[HASIL DETEKSI YOLOv8] {nama_penyakit} ({nilai_confidence})\n\n[REFERENSI]\n{combined_context}"
        
        payload_groq = {"model": "llama-3.1-8b-instant", "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}], "temperature": 0.2}
        res_groq = requests.post(url_groq, headers=headers_groq, json=payload_groq)
        hasil_interpretasi = res_groq.json()['choices'][0]['message']['content']
        
        return jsonify({
            "status": "success", "disease_name": nama_penyakit, "confidence": nilai_confidence,
            "retrieved_context": retrieved_contexts[0]["context"], "page": retrieved_contexts[0]["page"], "paragraph": retrieved_contexts[0]["paragraph"],
            "retrieved_contexts": retrieved_contexts, "interpretation": hasil_interpretasi
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==========================================
# ENDPOINT UTAMA: DETEKSI VIA FILE best.pt ASLI
# ==========================================
@app.route('/generate-interpretation', methods=['POST'])
def generate_interpretation():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "Format request salah. Butuh 'image'"}), 400
        
    image_file = request.files['image']
    lang = request.form.get('lang', 'id')
    
    try:
        # 1. Pastikan file best.pt sudah terdownload otomatis ke folder /tmp
        download_model_from_drive()
        
        # 2. Simpan gambar dari Android secara temporary di /tmp untuk dibaca YOLO
        temp_image_path = os.path.join("/tmp", image_file.filename)
        image_file.save(temp_image_path)
        
        # 3. Muat Model YOLOv8 menggunakan file best.pt asli
        model = YOLO(MODEL_PATH)
        
        # 4. Jalankan Prediksi Citra Mata (Threshold Confidence diatur ke 40%)
        results = model(temp_image_path, conf=0.40)
        
        # Hapus file gambar temporer segera agar tidak membebani storage server
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            
        # 5. Parsing hasil deteksi boks terkaya/tertinggi
        boxes = results[0].boxes
        if len(boxes) == 0:
            detected_disease = "normal"
            confidence_value = "100%"
        else:
            # Ambil boks dengan confidence tertinggi
            top_box = boxes[0]
            class_id = int(top_box.cls[0])
            detected_disease = model.names[class_id]
            confidence_value = f"{float(top_box.conf[0]) * 100:.2f}%"
            
        # 6. Alirkan hasil deteksi ke RAG Pinecone
        return process_eye_rag(nama_penyakit=detected_disease, nilai_confidence=confidence_value, lang=lang)
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Gagal menjalankan inferensi best.pt di server: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "active", "message": "Backend RAG Mata menggunakan Model Asli best.pt Aktif."})

if __name__ == '__main__':
    app.run(debug=True)
