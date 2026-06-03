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
PINECONE_HOST = os.environ.get("PINECONE_HOST")

# Daftar penyakit mata yang VALID & DIKUNCI sesuai fokus penelitian
VALID_DISEASES = {
    "normal",
    "katarak", "cataract",
    "konjungtivitis", "mata merah", "conjunctivitis",
    "uveitis", "radang uvea",
    "pterygium", "pterisium",
    "hordeolum", "bintitan", "stye"
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
    
    # Validasi input: Jika penyakit tidak terdaftar dalam ruang lingkup penelitian
    if clean_disease not in VALID_DISEASES:
        return jsonify({
            "status": "error",
            "message": f"Penyakit '{nama_penyakit}' di luar ruang lingkup cakupan penelitian mata."
        }), 400

    # Kondisi Khusus: Jika YOLOv8 mendeteksi mata "Normal"
    if clean_disease == "normal":
        msg_normal = (
            "Based on the external eye image analysis, your eye condition appears normal with no significant structural abnormalities detected."
            if lang.lower() == "en" else
            "Hasil analisis citra eksternal mata menunjukkan kondisi mata normal. Tidak ditemukan indikasi kelainan struktural atau tanda-tanda penyakit klinis."
        )
        return jsonify({
            "status": "success",
            "disease_name": "Normal",
            "confidence": nilai_confidence,
            "retrieved_context": "N/A (Patient is Normal)",
            "page": None,
            "paragraph": None,
            "retrieved_contexts": [],
            "interpretation": msg_normal
        }), 200

    try:
        # MAPPING METADATA KETAT
        disease_map = {
            "katarak": "katarak",
            "cataract": "katarak",
            "konjungtivitis": "konjungtivitis",
            "mata merah": "konjungtivitis",
            "conjunctivitis": "konjungtivitis",
            "uveitis": "uveitis",
            "radang uvea": "uveitis",
            "pterygium": "pterygium",
            "pterisium": "pterygium",
            "hordeolum": "hordeolum",
            "bintitan": "hordeolum",
            "stye": "hordeolum"
        }
        
        target_metadata_disease = disease_map.get(clean_disease, clean_disease)

        # 1. QUERY SEARCH KE PINECONE (MENGGUNAKAN INTEGRATED EMBEDDING CLOUD PINECONE)
        url_pinecone = f"{PINECONE_HOST}/query"
        headers_pc = {
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json"
        }
        
        query_internal = f"Clinical signs, symptoms, visual presentation, and medical treatment for {target_metadata_disease}."
        
        # Payload disesuaikan agar teks mentah diproses langsung oleh model embedding cloud Pinecone
        payload_pc = {
            "inputs": {"text": query_internal},  
            "topK": top_k,  
            "includeMetadata": True,
            "filter": {
                "disease": {"$eq": target_metadata_disease} 
            }
        }
        
        res_pc = requests.post(url_pinecone, headers=headers_pc, json=payload_pc)
        data_pc = res_pc.json()
        
        # Fallback nama kolom metadata sekunder jika Anda menggunakan nama "artifact"
        if "matches" not in data_pc or len(data_pc["matches"]) == 0:
            payload_pc["filter"] = {"artifact": {"$eq": target_metadata_disease}}
            res_pc = requests.post(url_pinecone, headers=headers_pc, json=payload_pc)
            data_pc = res_pc.json()
            
        # 2. PROSES EXTRACTION CONTEXT
        retrieved_contexts = []
        rank_counter = 1
        
        if "matches" in data_pc and len(data_pc["matches"]) > 0:
            for match in data_pc["matches"]:
                metadata = match.get("metadata", {})
                retrieved_contexts.append({
                    "rank": rank_counter,
                    "context": metadata.get("text", "Deskripsi medis tidak tersedia."),
                    "page": metadata.get("page", 1),
                    "paragraph": metadata.get("paragraph", 1),
                    "chunk_index": metadata.get("chunk_index", rank_counter),
                    "similarity_score": float(match.get("score", 0.0))
                })
                rank_counter += 1
        
        # Proteksi Fallback Code-Side (Bila database tidak merespon filter metadata secara tepat)
        if not retrieved_contexts:
            payload_back = {"inputs": {"text": query_internal}, "topK": top_k + 5, "includeMetadata": True}
            res_pc_back = requests.post(url_pinecone, headers=headers_pc, json=payload_back)
            data_pc_back = res_pc_back.json()
            
            if "matches" in data_pc_back:
                for match in data_pc_back["matches"]:
                    metadata = match.get("metadata", {})
                    metadata_text = metadata.get("text", "").lower()
                    
                    # Logika isolasi ketat antar kelas penyakit yang rentan tertukar
                    if target_metadata_disease == "konjungtivitis" and "hordeolum" in metadata_text and "konjungtivitis" not in metadata_text:
                        continue 
                    if target_metadata_disease == "hordeolum" and "conjunctivitis" in metadata_text and "hordeolum" not in metadata_text:
                        continue
                    if target_metadata_disease == "uveitis" and "konjungtivitis" in metadata_text and "uveitis" not in metadata_text:
                        continue
                        
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

        if not retrieved_contexts:
            retrieved_contexts.append({
                "rank": 1,
                "context": f"Gunakan pengetahuan medis umum terkait tata laksana penyakit {target_metadata_disease}.",
                "page": 1,
                "paragraph": 1,
                "chunk_index": 1,
                "similarity_score": 0.50
            })

        # 3. CONTEXT COMBINATION
        combined_context = "\n\n".join([
            f"[Referensi {item['rank']} | Halaman Buku {item['page']} | Paragraf {item['paragraph']}]\n{item['context']}"
            for item in retrieved_contexts
        ])

        # 4. SET UP BILINGUAL PROMPT DENGAN PROTEKSI FOKUS KELAS
        url_groq = "https://api.groq.com/openai/v1/chat/completions"
        headers_groq = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        if lang.lower() == "en":
            system_msg = """You are an expert AI Ophthalmology Assistant.
RULES:
- Provide a clinical interpretation strictly based on the provided Retrieved Contexts and YOLOv8 detection results.
- Maximum 2 paragraphs. Use professional and formal medical terminology.
- Explicitly pair lay terms and formal medical terms side by side (e.g., Stye/Hordeolum, Red Eye/Conjunctivitis, Cataract, Uveitis, or Pterygium)."""
            
            prompt = f"""[DETECTION RESULT]
Detected Disease: {nama_penyakit}
Confidence Score: {nilai_confidence}

[MEDICAL RETRIEVED CONTEXTS]
{combined_context}

TASK: Explain the condition, how it presents visually on eye images, and the recommended next clinical steps."""
        else:
            system_msg = """Anda adalah sistem AI Asisten Dokter Spesialis Mata yang sangat profesional namun ramah terhadap pasien awam.
ATURAN:
- Berikan interpretasi klinis dan langkah penanganan medis berdasarkan hasil deteksi sistem YOLOv8 dan dokumen referensi medis terpercaya yang disediakan (bisa dalam teks bahasa Indonesia maupun bahasa Inggris).
- Sampaikan jawaban dalam Bahasa Indonesia medis yang formal, jelas, dan terstruktur.
- Anda WAJIB menyandingkan istilah awam Indonesia dengan istilah medis resminya (contoh: Bintitan/Hordeolum, Mata Merah/Konjungtivitis, Katarak, Uveitis, atau Pterygium) agar pasien langsung paham.
- Jawaban maksimal 2 paragraf."""
            
            prompt = f"""[HASIL DETEKSI YOLOv8]
Penyakit Terdeteksi: {nama_penyakit}
Tingkat Keyakinan (Confidence): {nilai_confidence}

[DOKUMEN REFERENSI MEDIS (BILINGUAL INDO/ENG)]
{combined_context}

TUGAS: Jelaskan gambaran patologis penyakit tersebut pada citra mata, sebutkan nama awam beserta nama ilmiah medisnya, serta berikan rekomendasi tindakan medis awal atau rujukan yang harus dilakukan oleh pasien."""

        # 5. GENERATE INTERPRETASI VIA GROQ CLOUD
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

        # 6. JSON RETURN UNTUK FLUTTER / ANDROID STUDIO
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
# ENDPOINT POST & GET (UNTUK KELAS INFERENSI)
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

@app.route('/get-info', methods=['GET'])
def get_info():
    penyakit_query = request.args.get('penyakit', request.args.get('artefak', 'Katarak'))
    lang = request.args.get('lang', 'id')
    try:
        top_k = int(request.args.get('k', 3))
    except:
        top_k = 3
        
    return process_eye_rag(nama_penyakit=penyakit_query, nilai_confidence="98.50%", lang=lang, top_k=top_k)


# ==========================================
# ENDPOINT BARU: FITUR INTERAKTIF CHAT MATA
# ==========================================
@app.route('/chat', methods=['POST'])
def chat_mata():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"status": "error", "message": "Pesan dari user tidak boleh kosong."}), 400
    
    user_message = data.get('message')
    # Menerima rekaman riwayat percakapan dari aplikasi mobile
    chat_history = data.get('history', []) 
    lang = data.get('lang', 'id')
    
    # 1. Konfigurasi System Prompt Utama untuk Chatbot
    if lang.lower() == "en":
        system_msg = """You are a helpful, professional, and friendly AI Ophthalmology Assistant. 
        Answer the patient's questions clearly based on general medical eye health knowledge. 
        Keep your advice safe, responsible, and easy to understand."""
    else:
        system_msg = """Anda adalah AI Asisten Dokter Spesialis Mata yang ramah, sopan, dan profesional. 
        Jawablah pertanyaan pengguna atau pasien mengenai keluhan atau kesehatan mata mereka dengan penjelasan yang mudah dimengerti. 
        Gunakan bahasa Indonesia yang santun dan informatif. 
        Selalu ingatkan pasien untuk melakukan pemeriksaan langsung ke dokter spesialis mata jika gejala dirasa mengkhawatirkan atau memburuk."""

    # 2. Susun Struktur Chat Sesuai Format Groq/OpenAI (Memasukkan History)
    messages = [{"role": "system", "content": system_msg}]
    
    # Looping history yang dikirim dari mobile UI ke dalam pesan Chat Completions
    for chat in chat_history:
        messages.append({"role": chat['role'], "content": chat['content']})
        
    # Masukkan chat terupdate dari user di akhir array
    messages.append({"role": "user", "content": user_message})
    
    try:
        # 3. Tembak Data Ke API Groq
        url_groq = "https://api.groq.com/openai/v1/chat/completions"
        headers_groq = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload_groq = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "temperature": 0.4  
        }
        
        res_groq = requests.post(url_groq, headers=headers_groq, json=payload_groq)
        data_groq = res_groq.json()
        
        reply_ai = data_groq['choices'][0]['message']['content']
        
        return jsonify({
            "status": "success",
            "reply": reply_ai
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "active", "message": "Sistem RAG & Fitur Chat 6 Kelas Diagnosa Mata Aktif."})

if __name__ == '__main__':
    app.run(debug=True)
