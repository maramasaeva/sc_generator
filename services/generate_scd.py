# services/generate_scd.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def generate_supercollider_code(prompt):
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder", trust_remote_code=True)

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

    input_prompt = f"# Write a SuperCollider script based on the following description:\n# {prompt}\n\n(\n"

    outputs = generator(
        input_prompt,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = outputs[0]['generated_text']

    # Extract the code after the input prompt
    code = generated_text[len(input_prompt):]

    # Ensure code ends properly
    if not code.strip().endswith(');'):
        code += '\n);'

    return code