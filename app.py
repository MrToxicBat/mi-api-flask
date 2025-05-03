from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os

# Configura la clave de la API (asegúrate de definir GEMINI_API_KEY en Render)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

@app.route("/api/chat", methods=["POST"])
def chat():
    print("📥 Recibida petición POST")

    imagen = request.files.get("imagen")
    mensaje = request.form.get("mensaje", "")

    if not imagen:
        print("❌ No se recibió imagen en request.files")
        return jsonify({"respuesta": "⚠️ No se recibió imagen"})

    try:
        imagen_bytes = imagen.read()
        print(f"📸 Imagen recibida: {len(imagen_bytes)} bytes")
        print(f"🗨️ Mensaje recibido: {mensaje}")

        model = genai.GenerativeModel("gemini-pro-vision")
        respuesta = model.generate_content([
            mensaje,
            genai.Image.from_bytes(imagen_bytes)
        ])
        print("✅ Respuesta generada con éxito")

        return jsonify({"respuesta": respuesta.text})

    except Exception as e:
        print("🔥 Error al procesar:", str(e))
        return jsonify({"respuesta": f"❌ Error en servidor: {str(e)}"})
