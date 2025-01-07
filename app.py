'''
   Author  : Emon Hasan
   Email   : iconicemon01@gmail.com
   GitHub  : https://github.com/Md-Emon-Hasan
   LinkedIn: https://www.linkedin.com/in/emon-hasan/
   Date    : 01/07/2025
   Time    : 11:11
   Project : building won ai chatbot
   Purpose : building a chatbot uing openai and hugging face
'''

# Import required libraries
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template

from model.nlp_model import generate_response  # Import the NLP-related code

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)