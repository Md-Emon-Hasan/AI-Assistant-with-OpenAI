from transformers import TFAutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer
MODEL_NAME = "gpt2"  # Pretrained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Create text-generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="tf")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    user_input = request.json.get("prompt", "")
    if not user_input:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        # Generate response
        response = generator(user_input, max_length=150, num_return_sequences=1)
        generated_text = response[0]["generated_text"]
        return jsonify({"response": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
