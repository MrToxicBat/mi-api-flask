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

# ————— Instrucción del sistema —————
SYSTEM_INSTRUCTION = (
    "Eres una IA médica multimodal que ofrece un trato cálido y profesional. "
    "No utilices asteriscos en tu respuesta. "
    "Cuando el usuario envíe una imagen, reconoce que la tienes. "
    "Responde claramente al comando diagnóstico, resumen o tratamientos según lo solicite."
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
    logger.info(f"[analyze-image] SID={sid}, img length={len(img_b64)}")

    if img_b64:
        # Marcamos la imagen como recibida (aquí podrías guardar b64 o analizarla)
        state['image_desc'] = "Imagen recibida"
        logger.info(f"[analyze-image] SID={sid}, imagen marcada como recibida")
        response_text = "Imagen recibida correctamente."
    else:
        response_text = "No se ha enviado ninguna imagen."

    # Devolvemos siempre algo en `response`
    return jsonify({'response': response_text}), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    sid       = get_sid()
    state     = session_data.setdefault(sid, {"image_desc": ""})
    image_desc= state.get('image_desc','')

    data      = request.get_json() or {}
    user_text = data.get('message','').strip()

    logger.info(f"[chat] SID={sid}, image_desc='{image_desc}', user_text='{user_text}'")

    # Si no marcamos antes la imagen, pedimos que se envíe
    if not image_desc:
        return jsonify({'response':'Por favor, adjunta primero la imagen para analizarla.'}), 200

    # Construimos prompt definitivo
    prompt = (
        SYSTEM_INSTRUCTION + "\n\n"
        f"Imagen recibida: {image_desc}\n"
        f"Contexto: {user_text}\n\n"
        "Ahora genera diagnóstico, resumen y tratamientos según lo solicite el usuario."
    )
    logger.info(f"[chat] SID={sid}, prompt len={len(prompt)}")

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp  = model.generate_content({"parts":[{"text": prompt}]})
        ai_text = getattr(resp, "text","").strip()
        logger.info(f"[chat] SID={sid}, ai_text len={len(ai_text)}")
        return jsonify({'response': ai_text}), 200

    except Exception:
        logger.exception("[chat] Error generando respuesta IA")
        return jsonify({'response':'Lo siento, ha ocurrido un error interno.'}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
