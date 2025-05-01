import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import re

# --- Configuration ---
model_id = "01-ai/Yi-Coder-9B-Chat" # Instruction-tuned Gemma 2B

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for computation
)

# --- Load Tokenizer ---
print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Tokenizer loaded.")

# --- Load Model ---
print(f"Loading model {model_id} with 4-bit quantization...")
# Make sure a GPU is available
if torch.cuda.is_available():
    device = "cuda"
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    # Fallback to CPU - WARNING: This will be EXTREMELY slow
    device = "cpu"
    print("Warning: No GPU detected. Running on CPU will be very slow.")
    # Quantization might behave differently or not be supported well on CPU
    # For CPU only, you might remove quantization_config and hope it fits RAM

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config if device == "cuda" else None, # Only quantize if using GPU
    device_map="auto", # Automatically map model layers to available devices (GPU/CPU/RAM)
)
print("Model loaded.")

def generate_python_code(prompt_text, max_new_tokens=1024):
    """
    Generates Python code based on a prompt using the loaded model.

    Args:
        prompt_text (str): The instruction for code generation
                           (e.g., "Write a Python function to calculate factorial").
        max_new_tokens (int): Maximum number of tokens to generate for the response.

    Returns:
        str: The generated text, hopefully containing Python code.
    """
    if not prompt_text:
        return "Error: Please provide a prompt."

    # --- Prepare Prompt for Instruction-Tuned Gemma ---
    # Gemma uses a specific chat template. Using it helps the model understand roles.
    # We add a specific instruction to focus on Python code.
    full_prompt = f"Generate only the Python code for the following task, without any explanation before or after the code block:\n{prompt_text}"

    chat = [
        { "role": "user", "content": full_prompt },
    ]
    # apply_chat_template handles adding special tokens like <start_of_turn> etc.
    prompt_formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # --- Tokenize Input ---
    inputs = tokenizer(prompt_formatted, return_tensors="pt").to(device) # Move inputs to GPU if available

    # --- Generate Output ---
    print("\nGenerating code...")
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Optional parameters (can experiment with these):
            # do_sample=True,
            # temperature=0.7,
            # top_k=50,
            # top_p=0.95,
            pad_token_id=tokenizer.eos_token_id # Set pad token to EOS token for open-ended generation
        )
    print("Generation complete.")

    # --- Decode and Clean Output ---
    # Decode, skipping special tokens and the input prompt part
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the input prompt from the generated text
    # Find the end of the formatted prompt within the generated text
    prompt_end_index = generated_text.find(full_prompt)
    if prompt_end_index != -1:
        # Find the start of the model's response (might need adjustment based on model)
        # Gemma's template adds `<start_of_turn>model\n`
        model_response_start_marker = "<start_of_turn>model\n"
        response_start_index = generated_text.find(model_response_start_marker, prompt_end_index)
        if response_start_index != -1:
             # Get text after the marker
             code_output = generated_text[response_start_index + len(model_response_start_marker):].strip()
        else:
             # Fallback if marker not found exactly (less precise)
             code_output = generated_text[len(prompt_formatted):].strip() # Less reliable fallback
    else:
        code_output = "Error: Could not cleanly separate prompt from response." # Should not happen often

    # Optional: Try to extract only the code block if markdown ```python ... ``` is used
    match = re.search(r"```python\n(.*?)```", code_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # Return the whole cleaned output if no markdown block found
        return code_output

# Create a directory to store the generated code files
output_dir = "generated_code"
os.makedirs(output_dir, exist_ok=True)
print(f"Created directory '{output_dir}' for storing generated code.")

# Define the four prompts
predefined_prompts = [
    {
        "name": "prime_checker",
        "prompt": "Write a Python function that checks if a number is prime and includes a test case"
    },
    {
        "name": "data_visualizer",
        "prompt": "Create a Python script that generates and visualizes random data using matplotlib"
    },
    {
        "name": "file_processor", 
        "prompt": "Write a Python script that reads a text file, counts word frequency, and outputs the top 10 most common words"
    },
    {
        "name": "web_scraper",
        "prompt": "Create a simple web scraper using Python's requests and BeautifulSoup to extract headlines from a news website"
    }
]

# Generate code for each prompt and save to files
for item in predefined_prompts:
    print(f"\nGenerating code for: {item['name']}")
    
    code = generate_python_code(item['prompt'])
    
    # Save the generated code to a file
    file_path = os.path.join(output_dir, f"{item['name']}.py")
    
    with open(file_path, 'w') as file:
        file.write(f"# Generated from prompt: {item['prompt']}\n\n")
        file.write(code)
    
    print(f"Code saved to {file_path}")

print("\nAll code has been generated and saved to the '{output_dir}' directory.")
print("You can now execute these files directly.")