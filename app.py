import os, uuid, logging, base64
from flask import Flask, request, jsonify, session as flask_session
from flask_cors import CORS
import google.generativeai as genai

# ——— Configuración básica ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")
CORS(app, supports_credentials=True, origins=[os.getenv("FRONTEND_ORIGIN","https://code-soluction.com")])

# ——— Inicializar Gemini ———
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# … tus field_prompts y session_data igual que antes …

@app.route('/api/chat', methods=['POST'])
def chat():
    data       = request.json or {}
    user_text  = data.get('message','').strip()
    image_b64  = data.get('image')
    image_type = data.get('image_type')

    # gestión de sesión…
    session_id = flask_session.get('session_id')
    if not session_id or session_id not in session_data:
        session_id = str(uuid.uuid4())
        flask_session['session_id'] = session_id
        flask_session['step'] = 0
        session_data[session_id] = {}
    step = flask_session['step']
    collected = session_data[session_id]

    # Montar inputs
    inputs = []
    if image_b64 and not user_text:
        inputs.append({"image":{"data":image_b64,"mime_type":image_type}})
        inputs.append({"text":"Por favor, analiza esta imagen médica."})
    else:
        # tu lógica de recolección de campos…
        # al final siempre acabas con un `prompt` de texto
        inputs.append({"text": get_system_instruction()})
        inputs.append({"text": prompt})

    # ——  Aquí es donde cambias el nombre del modelo ——
    model = genai.GenerativeModel("models/gemini-1.5-multimodal-preview")

    try:
        resp = model.generate_content(inputs)
        ai_text = getattr(resp, "text","").strip()
        return jsonify({"response": ai_text})
    except Exception as e:
        logger.error("Error en /api/chat:", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT",5000)))
