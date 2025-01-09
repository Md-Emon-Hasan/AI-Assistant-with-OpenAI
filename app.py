import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# Ensure TensorFlow uses only the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.experimental.set_visible_devices([], "GPU")

# Limit memory usage
physical_devices = tf.config.experimental.list_physical_devices('CPU')
for device in physical_devices:
    tf.config.experimental.set_virtual_device_configuration(
        device,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)]
    )

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    user_input = request.json.get("prompt", "")
    if not user_input:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        from model.nlp_model import generate_response  # Import here to avoid initialization overhead
        generated_text = generate_response(user_input)
        return jsonify({"response": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
