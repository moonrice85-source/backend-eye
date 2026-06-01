import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==========================================
# CONFIGURATION & CLOUD CREDENTIALS
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

NAMA_INDEX = "vector-index-my"
PINECONE_HOST = "https://vector-index-my-oxdsd3s.svc.aped-4627-b74a.pinecone.io"


# Daftar penyakit mata yang valid dalam cakupan penelitian Anda
VALID_DISEASES = {
    "normal", 
    "katarak", 
    "hordeolum", 
    "conjunctivitis", 
    "pterygium",
    "uveitis"
}

def normalize_label(text: str) -> str:
    return " ".join(
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("'", " ")
        .replace("’", " ").split()
    )

# ==========================================
# LOGIKA INTI: INTEGRASI RAG MATA (REST API MURNI)
# ==========================================
def process_eye_rag(nama_penyakit: str, nilai_confidence: str, lang: str, top_k: int = 3):
    clean_disease = normalize_label(nama_penyakit)
    
    # Kondisi Khusus: Jika YOLOv8 mendeteksi mata "Normal"
    if clean_disease == "normal":
        msg_normal = (
            "Based on the external eye image analysis, your eye condition appears normal with no significant structural abnormalities detected."
            if lang.lower() == "en" else
            "Hasil analisis citra eksternal mata menunjukkan kondisi mata normal. Tidak ditemukan indikasi kelainan struktural atau tanda-tanda penyakit klinis."
        )
        return jsonify({
            "status": "success",
            "disease_name": nama_penyakit,
            "confidence": nilai_confidence,
            "retrieved_context": "N/A (Patient is Normal)",
            "page": None,
            "paragraph": None,
            "retrieved_contexts": [],
            "interpretation": msg_normal
        }), 200

    try:
        # 1. QUERY SEARCH KE PINECONE SERVERLESS INFERENCE (Menggunakan Pencarian Teks Langsung)
        url_pinecone = f"{PINECONE_HOST}/query"
        headers_pc = {
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json"
        }
        
        # Buat query penjelas otomatis untuk mengambil referensi klinis terdekat
        query_internal = f"Gejala, tanda klinis pada citra fundus/eksternal, dan penanganan medis untuk penyakit {clean_disease}."
        
        payload_pc = {
            "inputs": query_internal,
            "topK": max(top_k * 4, 10),  # Mengambil kandidat ekstra untuk penyaringan metadata penyakit
            "includeMetadata": True
        }
        
        res_pc = requests.post(url_pinecone, headers=headers_pc, json=payload_pc)
        data_pc = res_pc.json()
        
        # 2. FILTERING METADATA PENYAKIT (SAMA PERSIS DENGAN STANDAR KODE FASTAPI ANDA)
        retrieved_contexts = []
        rank_counter = 1
        
        if "matches" in data_pc:
            for match in data_pc["matches"]:
                metadata = match.get("metadata", {})
                # Filter ketat: Memastikan potongan dokumen medis cocok dengan kelas penyakit hasil deteksi YOLOv8
                metadata_disease = metadata.get("disease", metadata.get("artifact", "")).lower()
                
                if metadata_disease == clean_disease or clean_disease in normalize_label(metadata.get("text", "")):
                    retrieved_contexts.append({
                        "rank": rank_counter,
                        "context": metadata.get("text", "Deskripsi medis tidak tersedia."),
                        "page": metadata.get("page", 1),
                        "paragraph": metadata.get("paragraph", 1),
                        "chunk_index": metadata.get("chunk_index", rank_counter),
                        "similarity_score": float(match.get("score", 0.0))
                    })
                    rank_counter += 1
                
                if len(retrieved_contexts) >= top_k:
                    break

        # Fallback jika tidak ada chunk yang lolos filter metadata
        if not retrieved_contexts:
            retrieved_contexts.append({
                "rank": 1,
                "context": f"Gunakan pengetahuan medis umum terkait tata laksana penyakit {nama_penyakit}.",
                "page": 1,
                "paragraph": 1,
                "chunk_index": 1,
                "similarity_score": 0.50
            })

        # 3. PENGGABUNGAN KONTEKS UNTUK PROMPT LLM
        combined_context = "\n\n".join([
            f"[Referensi {item['rank']} | Halaman Buku {item['page']} | Paragraf {item['paragraph']}]\n{item['context']}"
            for item in retrieved_contexts
        ])

        # 4. SYSTEM PROMPT & USER PROMPT BERDASARKAN BAHASA
        if lang.lower() == "en":
            system_msg = """You are an expert AI Ophthalmologist Assistant.
RULES:
- Provide a clinical interpretation strictly based on the provided Retrieved Contexts and YOLOv8 detection results.
- Maximum 2 paragraphs. Use professional and formal medical terminology.
- If the context is completely irrelevant, rely on standard ophthalmology guidelines for the detected disease."""
            
            prompt = f"""[DETECTION RESULT]
Detected Disease: {nama_penyakit}
Confidence Score: {nilai_confidence}

[MEDICAL RETRIEVED CONTEXTS]
{combined_context}

TASK: Explain the condition, how it presents visually on eye images, and the recommended next clinical steps."""
        else:
            system_msg = """Anda adalah sistem AI Asisten Dokter Spesialis Mata yang sangat profesional.
ATURAN:
- Berikan interpretasi klinis dan langkah penanganan medis berdasarkan hasil deteksi sistem YOLOv8 dan dokumen referensi medis terpercaya yang disediakan.
- Sampaikan jawaban dalam Bahasa Indonesia medis yang formal, jelas, dan terstruktur.
- Jawaban maksimal 2 paragraf."""
            
            prompt = f"""[HASIL DETEKSI YOLOv8]
Penyakit Terdeteksi: {nama_penyakit}
Tingkat Keyakinan (Confidence): {nilai_confidence}

[DOKUMEN REFERENSI MEDIS]
{combined_context}

TUGAS: Jelaskan gambaran patologis penyakit tersebut pada citra mata, serta berikan rekomendasi tindakan medis awal atau rujukan yang harus dilakukan oleh pasien."""

        # 5. GENERATE GENERASI TEKS VIA GROQ CLOUD API (Llama 3.1)
        url_groq = "https://api.groq.com/openai/v1/chat/completions"
        headers_groq = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload_groq = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        
        res_groq = requests.post(url_groq, headers=headers_groq, json=payload_groq)
        data_groq = res_groq.json()
        
        hasil_interpretasi = data_groq['choices'][0]['message']['content']
        first_context = retrieved_contexts[0]

        # 6. JSON RETURN (SAMA PERSIS DENGAN SKEMA OUTPUT FASTAPI TA ANDA)
        return jsonify({
            "disease_name": nama_penyakit,
            "confidence": nilai_confidence,
            "top_k": top_k,
            "retrieved_context": first_context["context"],
            "page": first_context["page"],
            "paragraph": first_context["paragraph"],
            "chunk_index": first_context["chunk_index"],
            "similarity_score": first_context["similarity_score"],
            "retrieved_contexts": retrieved_contexts,
            "interpretation": hasil_interpretasi
        }), 200

    except Exception as e:
        error_msg = str(e)
        if 'data_groq' in locals():
            error_msg += f" | Detail Groq: {data_groq}"
        return jsonify({"status": "error", "message": error_msg}), 500

# ==========================================
# ROUTE 1: UNTUK KONEKSI ANDROID STUDIO (POST)
# ==========================================
@app.route('/generate-interpretation', methods=['POST'])
def generate_interpretation():
    data = request.json
    if not data or 'penyakit' not in data or 'confidence' not in data:
        return jsonify({"status": "error", "message": "Format request salah. Butuh 'penyakit' dan 'confidence'"}), 400
    
    lang = data.get('lang', 'id')
    try:
        top_k = int(data.get('k', 3))
    except:
        top_k = 3
        
    return process_eye_rag(nama_penyakit=data['penyakit'], nilai_confidence=data['confidence'], lang=lang, top_k=top_k)

# ==========================================
# ROUTE 2: UNTUK UJI COBA INSTAN VIA BROWSER (GET)
# ==========================================
@app.route('/get-info', methods=['GET'])
def get_info():
    # Mengambil parameter dari URL query string (?artefak=Katarak atau ?penyakit=Katarak)
    penyakit_query = request.args.get('penyakit', request.args.get('artefak', 'Katarak'))
    lang = request.args.get('lang', 'id')
    try:
        top_k = int(request.args.get('k', 3))
    except:
        top_k = 3
        
    return process_eye_rag(nama_penyakit=penyakit_query, nilai_confidence="98.50%", lang=lang, top_k=top_k)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "active", "message": "Sistem RAG Diagnosa Penyakit Mata Aktif (Serverless Mode)."})
