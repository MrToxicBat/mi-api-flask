import os
import uuid
import logging
from flask import Flask, request, jsonify, make_response, session as flask_session
from flask_cors import CORS
import google.generativeai as genai

# ——— Configuración básica ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# Habilitar CORS
CORS(app,
     supports_credentials=True,
     resources={r"/api/*": {"origins": ["https://code-soluction.com"]}})

# Inicializar Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "models/gemini-2.0-flash"

# ——— Prompt base análisis ———
def get_system_instruction_analysis():
    return (
        "Eres un médico radiólogo experto. Analiza la imagen médica y describe lo que ves "
        "en lenguaje claro pero profesional. Explica posibles diagnósticos y observaciones importantes."
    )

# ——— Prompt base conversación posterior ———
def get_system_instruction_conversation():
    return (
        "Eres un médico traumatólogo conversando con un paciente sobre un diagnóstico previo. "
        "Responde de forma cercana, empática y explicativa, usando un lenguaje que pueda entender una persona sin conocimientos médicos. "
        "Incluye recomendaciones prácticas, opciones de tratamiento y posibles pasos a seguir, sin sonar robótico."
    )

# ——— Memoria de sesión ———
# Estructura: {sid: {"analysis": str}}
session_data = {}

# ——— Análisis de imagen ———
@app.route('/api/analyze-image', methods=['OPTIONS', 'POST'])
def analyze_image():
    if request.method == 'OPTIONS':
        return make_response()
    data = request.json or {}
    image_b64 = data.get('image')
    image_type = data.get('image_type')

    if not image_b64 or not image_type:
        return jsonify({"error": "Falta image o image_type"}), 400

    # Nueva sesión
    sid = str(uuid.uuid4())
    flask_session['session_id'] = sid
    session_data[sid] = {"analysis": None}

    prompt = (
        "🖼️ Análisis de imagen médica:\n"
        "1. Describe hallazgos relevantes.\n"
        "2. Indica posible diagnóstico.\n"
        "3. Añade recomendaciones iniciales.\n"
        "Usa un tono profesional pero comprensible."
    )

    parts = [
        {"mime_type": image_type, "data": image_b64},
        {"text": f"{get_system_instruction_analysis()}\n\n{prompt}"}
    ]

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content({"parts": parts})
        analysis_text = resp.text.strip()

        # Guardar análisis en sesión
        session_data[sid]["analysis"] = analysis_text

        return jsonify({
            "response": analysis_text,
            "session_id": sid
        })
    except Exception as e:
        logger.error("Error en /api/analyze-image", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ——— Conversación posterior ———
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_text = data.get('message', '').strip()
    sid = data.get('session_id') or flask_session.get('session_id')

    if not sid or sid not in session_data:
        return jsonify({"error": "No hay análisis previo en la sesión"}), 400

    analysis_context = session_data[sid].get("analysis", "")

    # Construir prompt con contexto del análisis
    prompt_text = (
        f"Diagnóstico previo:\n{analysis_context}\n\n"
        f"Pregunta del paciente: {user_text}\n\n"
        "Responde en un tono empático y explicativo."
    )

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content({
            "parts": [
                {"text": f"{get_system_instruction_conversation()}\n\n{prompt_text}"}
            ]
        })
        return jsonify({"response": resp.text.strip()})
    except Exception as e:
        logger.error("Error en /api/chat", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
