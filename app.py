import os
import uuid
import logging
from flask import Flask, request, jsonify, session as flask_session
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
    resources={r"/api/*": {
        "origins": [
            "https://code-soluction.com",
            "https://www.code-soluction.com",
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ]
    }},
    supports_credentials=True
)

# ————— Inicializar Gemini —————
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "models/gemini-2.0-flash"

# ————— Estado en memoria (desarrollo) —————
session_data = {}

# ————— Instrucción del sistema —————
SYSTEM_INSTRUCTION = (
    "Eres una IA médica multimodal que ofrece un trato cálido y profesional. "
    "No utilices asteriscos en tu respuesta. "
    "Cuando el usuario envíe una imagen adjunta, descríbela con detalle; cuando envíe un contexto, entiéndelo perfectamente. "
    "Responde claramente al comando \"diagnóstico\", \"resumen\" o \"tratamientos\" según lo solicite el usuario."
)

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    data    = request.get_json() or {}
    img_b64 = data.get('image','')
    # Aseguramos ID de sesión
    sid = flask_session.get('session_id') or str(uuid.uuid4())
    flask_session['session_id'] = sid

    state = session_data.setdefault(sid, {
        "welcomed": False,
        "image_desc": None
    })

    description = ""
    if img_b64:
        try:
            resp = genai.annotate_image(
                model="models/gemini-image-alpha",
                image=img_b64,
                supports=["TEXT"]
            )
            if getattr(resp, "annotations", None):
                description = resp.annotations[0].text or ""
            logger.info(f"[analyze-image] Descripción: {description[:80]}…")
        except Exception:
            logger.exception("[analyze-image] Error llamando a Gemini")
    else:
        logger.warning("[analyze-image] No se proporcionó imagen")

    # Guardamos descripción en el estado
    state['image_desc'] = description

    # Respondemos en el campo `response` para que el front siga usando data.response
    return jsonify({'response': description}), 200


@app.route('/api/chat', methods=['POST'])
def chat():
    data      = request.get_json() or {}
    user_text = data.get('message','').strip()

    # ID de sesión
    sid = flask_session.get('session_id') or str(uuid.uuid4())
    flask_session['session_id'] = sid

    state = session_data.setdefault(sid, {
        "welcomed": False,
        "image_desc": None
    })

    # 1) Bienvenida inicial
    if not state['welcomed']:
        state['welcomed'] = True
        welcome = (
            "👋 Bienvenido al asistente médico. "
            "Por favor, adjunta la imagen relevante y luego proporciona un breve contexto de lo ocurrido."
        )
        return jsonify({'response': welcome}), 200

    # 2) Si no hay descripción de imagen, la solicitamos
    if not state.get('image_desc'):
        ask_img = "Aún no he recibido una imagen. Adjunta la imagen médica para que pueda analizarla."
        return jsonify({'response': ask_img}), 200

    # 3) Ya tenemos imagen + contexto → construimos prompt
    prompt = (
        SYSTEM_INSTRUCTION + "\n\n" +
        "Imagen descrita: " + state['image_desc'] + "\n" +
        "Contexto: " + user_text + "\n\n" +
        "Por favor, proporciona un diagnóstico detallado, un resumen de tu análisis y, si lo solicito, tratamientos adecuados."
    )
    logger.info(f"[chat] Prompt a Gemini: {prompt[:100]}…")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp  = model.generate_content({"parts": [{"text": prompt}]})
        ai_text = getattr(resp, "text", "").strip()
        logger.info(f"[chat] Respuesta IA: {ai_text[:80]}…")

        return jsonify({"response": ai_text}), 200

    except Exception:
        logger.exception("[chat] Error generando respuesta IA")
        return jsonify({"response": "Lo siento, ha ocurrido un error interno."}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
