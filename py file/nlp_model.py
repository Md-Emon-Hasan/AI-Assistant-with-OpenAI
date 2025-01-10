from transformers import TFAutoModelForCausalLM, AutoTokenizer, pipeline

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
