import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


model_id = "01-ai/Yi-Coder-9B-Chat" # Instruction-tuned Gemma 2B


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for computation
)

print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Tokenizer loaded.")

print(f"Loading model {model_id} with 4-bit quantization...")
# Make sure a GPU is available
if torch.cuda.is_available():
    device = "cuda"
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Warning: No GPU detected. Running on CPU will be very slow.")
    

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config if device == "cuda" else None, 
    device_map="auto",
)
print("Model loaded.")