from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os

# Configura tu API key desde variables de entorno
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

@app.route("/api/chat", methods=["POST"])
def chat():
    imagen = request.files.get("imagen")
    mensaje = request.form.get("mensaje", "")

    if not imagen:
        return jsonify({"respuesta": "⚠️ No se recibió imagen"})

    try:
        imagen_bytes = imagen.read()

        # Usa el modelo actualizado compatible con imágenes
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content([
            {"text": mensaje},
            {"mime_type": "image/jpeg", "data": imagen_bytes}
        ])

        return jsonify({"respuesta": response.text})

    except Exception as e:
        return jsonify({"respuesta": f"❌ Error al procesar: {str(e)}"})
