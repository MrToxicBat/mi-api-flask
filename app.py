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

# ————— Estado en memoria —————
session_data = {}

# ————— Instrucción del sistema —————
SYSTEM_INSTRUCTION = (
    "Eres una IA médica multimodal que ofrece un trato cálido y profesional. "
    "No utilices asteriscos en tu respuesta. "
    "Cuando el usuario envíe una imagen adjunta, descríbela con detalle; "
    "cuando envíe un contexto, entiéndelo perfectamente. "
    "Responde claramente al comando diagnóstico, resumen o tratamientos según lo solicite el usuario."
)

def get_sid():
    sid = request.headers.get('X-Session-Id')
    return sid or str(uuid.uuid4())

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    sid = get_sid()
    state = session_data.setdefault(sid, {"image_desc": ""})

    data    = request.get_json() or {}
    img_b64 = data.get('image','')
    logger.info(f"[analyze-image] SID={sid}, received image length={len(img_b64)}")

    # -- No intentamos más llamar a annotate_image que falla --
    description = ""
    if img_b64:
        logger.warning("[analyze-image] el SDK no soporta annotate_image; devolviendo descripción vacía")
    else:
        logger.warning("[analyze-image] No se proporcionó imagen")

    state['image_desc'] = description

    # Devolvemos siempre en `response` para encajar con el front
    return jsonify({'response': description}), 200


@app.route('/api/chat', methods=['POST'])
def chat():
    sid = get_sid()
    state = session_data.setdefault(sid, {"image_desc": ""})

    data      = request.get_json() or {}
    user_text = data.get('message','').strip()
    img_desc  = state.get('image_desc','')

    logger.info(f"[chat] SID={sid}, user_text={user_text!r}, img_desc_len={len(img_desc)}")

    # Si no hay imagen, pedimos primero la imagen
    if not img_desc:
        return jsonify({'response':'Por favor, adjunta primero la imagen para analizarla.'}), 200

    # Construimos el prompt final
    prompt = (
        SYSTEM_INSTRUCTION + "\n\n"
        f"Imagen descrita: {img_desc}\n"
        f"Contexto: {user_text}\n\n"
        "Ahora genera diagnóstico, resumen y tratamientos según pida el usuario."
    )
    logger.info(f"[chat] SID={sid}, prompt len={len(prompt)}")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp  = model.generate_content({"parts":[{"text":prompt}]})
        ai_text = getattr(resp,"text","").strip()
        logger.info(f"[chat] SID={sid}, ai_text len={len(ai_text)}")
        return jsonify({'response':ai_text}), 200

    except Exception:
        logger.exception("[chat] Error generando respuesta IA")
        return jsonify({'response':'Lo siento, ha ocurrido un error interno.'}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT",5000))
    app.run(host='0.0.0.0',port=port)
