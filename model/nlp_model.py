'''
   Author  : Emon Hasan
   Email   : iconicemon01@gmail.com
   GitHub  : https://github.com/Md-Emon-Hasan
   LinkedIn: https://www.linkedin.com/in/emon-hasan/
   Date    : 01/07/2025
   Time    : 09:11
   Project : building won ai chatbot
   Purpose : building a chatbot uing openai and hugging face
'''

# Import required libraries
from transformers import TFAutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline

# Load model and tokenizer
MODEL_NAME = "gpt2"  # Pretrained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Create text-generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="tf")

def generate_response(user_input: str) -> str:
    try:
        # Generate response
        response = generator(user_input, max_length=150, num_return_sequences=1)
        generated_text = response[0]["generated_text"]
        return generated_text
    except Exception as e:
        raise ValueError(f"Error during text generation: {str(e)}")
