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
VISION_MODEL = "gemini-pro-vision"  # Modelo especializado para imágenes

# ————— Estado en memoria —————
session_data = {}

# ————— Instrucción del sistema para análisis de imagen —————
IMAGE_INSTRUCTION = (
    "Eres una IA médica multimodal. Has recibido una imagen médica. "
    "Sin pedir más contexto, analiza la imagen y responde con las siguientes secciones separadas y claras: "
    "1. Resumen\n2. Hallazgos relevantes\n3. Posibles diagnósticos\n4. Recomendaciones"
)

# ————— Helper para SID —————
def get_sid():
    sid = request.headers.get('X-Session-Id')
    return sid or str(uuid.uuid4())

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    sid = get_sid()
    session_data.setdefault(sid, {})

    data = request.get_json() or {}
    img_b64 = data.get('image', '')
    logger.info(f"[analyze-image] SID={sid}, img length={len(img_b64)}")

    if not img_b64:
        return jsonify({'response': 'No se ha recibido ninguna imagen.'}), 400

    try:
        # Configurar el modelo de visión
        model = genai.GenerativeModel(VISION_MODEL)
        
        # Preparar los componentes del mensaje
        image_part = {
            "mime_type": data.get('image_type', 'image/png'),
            "data": img_b64
        }
        
        # Generar la respuesta
        response = model.generate_content(
            contents=[IMAGE_INSTRUCTION, image_part]
        )
        
        # Manejar la respuesta
        if not response.text:
            raise ValueError("La API no devolvió una respuesta válida")
            
        ai_text = response.text
        logger.info(f"[analyze-image] SID={sid}, respuesta exitosa")
        
        return jsonify({
            'response': ai_text.strip(),
            'status': 'success'
        }), 200

    except Exception as e:
        logger.error(f"[analyze-image] Error: {str(e)}", exc_info=True)
        return jsonify({
            'response': 'Error al procesar la imagen. Por favor, inténtalo de nuevo.',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'response': 'El mensaje no puede estar vacío'}), 400
            
        # Implementación básica de chat (opcional)
        return jsonify({
            'response': 'Actualmente solo soportamos análisis de imágenes',
            'status': 'info'
        }), 200
        
    except Exception as e:
        logger.error(f"[chat] Error: {str(e)}")
        return jsonify({
            'response': 'Error en el servidor',
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': VISION_MODEL,
        'ready': True
    }), 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv("FLASK_DEBUG", False))
