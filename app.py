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

# Prompt único para todo el flujo
UNIFIED_PROMPT = """
Eres una inteligencia artificial médica especializada en el análisis de imágenes médicas 
(radiografías, resonancias, tomografías) y en responder consultas médicas por texto. 
Debes actuar como un médico especialista, con explicaciones claras, precisas y comprensibles 
para el paciente, pero con rigor técnico.

Instrucciones:
- Si el usuario envía una imagen médica, analiza posibles patologías, anomalías o hallazgos relevantes.
- Si el usuario envía texto, responde de forma clara, con un diagnóstico probable o sugerencias basadas en los datos.
- Explica términos médicos complejos de forma sencilla para que el paciente los entienda.
- Si hay varias posibilidades diagnósticas, indícalas y explica la diferencia.
- Nunca inventes información médica; si no tienes certeza, indícalo y sugiere consulta con un especialista.
- Usa un tono profesional y empático.
- Si el contenido no es médico, indícalo amablemente.

Formato de respuesta:
1. Resumen breve del hallazgo o respuesta.
2. Explicación detallada.
3. Sugerencias o próximos pasos.

Ejemplo de respuesta para imagen:
Resumen: Posible fractura distal del radio.
Explicación: En la imagen se observa una línea radiolúcida que atraviesa la región distal...
Sugerencias: Recomiendo inmovilización y valoración por traumatología.

Ejemplo de respuesta para texto:
Resumen: Posible infección respiratoria.
Explicación: Por los síntomas de fiebre, tos productiva y dolor torácico...
Sugerencias: Acudir a consulta médica para revisión y posible tratamiento antibiótico.

Responde siempre siguiendo el formato anterior.
"""

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
        "🖼️ Análisis de imagen médica:\n"
        "Por favor analiza esta imagen y responde siguiendo las instrucciones generales."
    )

    parts = [
        {"mime_type": image_type, "data": image_b64},
        {"text": f"{UNIFIED_PROMPT}\n\n{prompt}"}
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

    # Construir prompt
    if analysis_context:
        prompt_text = (
            f"Diagnóstico previo:\n{analysis_context}\n\n"
            f"Pregunta del paciente: {user_text}\n\n"
            "Responde siguiendo las instrucciones generales, en un tono empático y explicativo."
        )
    else:
        prompt_text = (
            f"Pregunta del paciente: {user_text}\n\n"
            "Responde siguiendo las instrucciones generales, en un tono empático y explicativo."
        )

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(f"{UNIFIED_PROMPT}\n\n{prompt_text}")
        return jsonify({"response": resp.text.strip()})
    except Exception as e:
        logger.error("Error en /api/chat", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
