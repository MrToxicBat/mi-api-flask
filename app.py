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

# Habilitar CORS para /api/*
CORS(app,
     supports_credentials=True,
     resources={r"/api/*": {"origins": ["https://code-soluction.com"]}})

# Inicializar Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "models/gemini-2.0-flash"

# Prompt base para conversación humana con estructura resumida, diagnóstico y tratamiento
def build_prompt_human_dialog(user_text: str, analysis_context: str) -> str:
    prompt = f"""
Eres un médico especialista que conversa con un paciente. Tu objetivo es responder de forma humana, empática y cercana, como si estuvieras hablando con un amigo que quiere entender su situación médica.

Cuando respondas, siempre estructura tu respuesta de forma natural integrando estas tres partes clave:

1. Resumen: Una explicación sencilla y breve de la situación actual o hallazgos.
2. Diagnóstico: Qué significa esa situación en términos médicos, explicado con palabras fáciles.
3. Tratamientos: Qué opciones tiene el paciente, recomendaciones prácticas y pasos a seguir.

No hagas listas rígidas ni uses lenguaje técnico complicado. Usa un tono cálido, positivo y claro. Puedes empezar saludando o agradeciendo la confianza.

Recuerda siempre ser humano, claro y paciente. No repitas lo que dice el paciente textualmente ni respondas de forma mecánica.

Pregunta del paciente: {user_text}

Información clínica previa:
{analysis_context}
"""
    return prompt

# Memoria de sesión
session_data = {}

@app.route('/api/analyze-image', methods=['OPTIONS', 'POST'])
def analyze_image():
    if request.method == 'OPTIONS':
        return make_response()
    data = request.json or {}
    image_b64 = data.get('image')
    image_type = data.get('image_type')

    if not image_b64 or not image_type:
        return jsonify({"error": "Falta image o image_type"}), 400

    # Crear nueva sesión
    sid = str(uuid.uuid4())
    flask_session['session_id'] = sid
    session_data[sid] = {"analysis": None}

    prompt = (
        "Por favor analiza esta imagen médica y describe lo que ves en un lenguaje claro y profesional, "
        "incluyendo posibles diagnósticos y recomendaciones iniciales."
    )

    parts = [
        {"mime_type": image_type, "data": image_b64},
        {"text": prompt}
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

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_text = data.get('message', '').strip()
    sid = data.get('session_id') or flask_session.get('session_id')

    analysis_context = ""
    if sid and sid in session_data:
        analysis_context = session_data[sid].get("analysis", "")

    # Construir prompt usando la función para diálogo humano y estructura clara
    prompt_text = build_prompt_human_dialog(user_text, analysis_context)

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content({"parts": [{"text": prompt_text}]})
        return jsonify({"response": resp.text.strip()})
    except Exception as e:
        logger.error("Error en /api/chat", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
