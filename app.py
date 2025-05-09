from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

@app.route("/api/chat", methods=["POST"])
def chat():
    imagen = request.files.get("imagen")
    mensaje = request.form.get("mensaje", "")

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        parts = []
        if mensaje:
            parts.append({"text": mensaje})
        if imagen:
            imagen_bytes = imagen.read()
            parts.append({
                "mime_type": imagen.mimetype,
                "data": imagen_bytes
            })

        if not parts:
            return jsonify({"respuesta": "⚠️ No se recibió entrada válida"})

        response = model.generate_content(parts)

        return jsonify({"respuesta": response.text})

    except Exception as e:
        return jsonify({"respuesta": f"❌ Error al procesar: {str(e)}"})
