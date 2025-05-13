import os
import uuid
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# ————— Configuración básica —————
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")

# ————— CORS —————
CORS(
    app,
    resources={r"/api/*": {"origins": [
        "https://code-soluction.com",
        "https://www.code-soluction.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]}},
    supports_credentials=True
)

# ————— Inicializar Gemini —————
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "models/gemini-2.0-flash"

# ————— Estado en memoria —————
session_data = {}

# ————— Instrucción del sistema para análisis de imagen —————
IMAGE_INSTRUCTION = (
    "Eres una IA médica multimodal. Has recibido una imagen médica. "
    "Sin solicitar más contexto, analiza la imagen y proporciona 1) Un resumen de lo que ves, 2) Un diagnóstico preliminar, y 3) Tratamientos sugeridos. "
    "No uses asteriscos y sepáralo en secciones claras bajo los títulos: Resumen, Diagnóstico y Tratamientos."
)

# ————— Helper para SID —————
def get_sid():
    sid = request.headers.get('X-Session-Id')
    return sid or str(uuid.uuid4())

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    sid = get_sid()
    session_data.setdefault(sid, {})

    data    = request.get_json() or {}
    img_b64 = data.get('image', '')
    logger.info(f"[analyze-image] SID={sid}, img length={len(img_b64)}")

    if not img_b64:
        return jsonify({'response': 'No se ha recibido ninguna imagen.'}), 200

    # Construir prompt usando solo la imagen
    prompt = IMAGE_INSTRUCTION + "\n\n" + "Imagen (base64): " + img_b64[:1000]
    logger.info(f"[analyze-image] SID={sid}, prompt len={len(prompt)}")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content({"parts": [{"text": prompt}]})
        ai_text = getattr(resp, "text", "").strip()
        logger.info(f"[analyze-image] SID={sid}, ai_text len={len(ai_text)}")
        return jsonify({'response': ai_text}), 200
    except Exception:
        logger.exception("[analyze-image] Error generando análisis")
        return jsonify({'response': 'Lo siento, no pude procesar la imagen.'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    # Este endpoint queda para contextos adicionales, pero vacío si no se usa
    return jsonify({'response': 'Endpoint de chat deshabilitado cuando se analiza imagen directamente.'}), 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
