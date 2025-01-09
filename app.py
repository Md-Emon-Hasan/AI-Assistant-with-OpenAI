import os
import tensorflow as tf

# Set environment variable to disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs (only errors are shown)

# Optional: Set TensorFlow logger to only show errors
tf.get_logger().setLevel('ERROR')

# Check if TensorFlow is detecting a GPU (It should not)
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU, which is not expected as we disabled it.")
else:
    print("TensorFlow is using the CPU as expected.")

# Import required libraries
from transformers import TFAutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify, render_template

# Load model and tokenizer
MODEL_NAME = "gpt2"  # Pretrained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Create text-generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="tf")

# Function to generate a response using the model
def generate_response(user_input: str) -> str:
    try:
        # Generate response
        response = generator(user_input, max_length=150, num_return_sequences=1)
        generated_text = response[0]["generated_text"]
        return generated_text
    except Exception as e:
        raise ValueError(f"Error during text generation: {str(e)}")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # You can return HTML or templates here if needed

@app.route("/generate", methods=["POST"])
def generate():
    user_input = request.json.get("prompt", "")
    if not user_input:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        # Use the generate_response function to generate text
        generated_text = generate_response(user_input)
        return jsonify({"response": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
