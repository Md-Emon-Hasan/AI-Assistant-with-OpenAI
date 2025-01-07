from flask import Flask, request, jsonify, render_template
from nlp_model import generate_response  # Import the NLP-related code

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
        # Use the generate_response function from nlp_model.py
        generated_text = generate_response(user_input)
        return jsonify({"response": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
