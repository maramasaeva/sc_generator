import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

class LLMService:
    def __init__(self, model_name="Salesforce/codegen-2B"):
        print("Loading model and tokenizer...")
        login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token)

    def generate_modified_scd_code(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate output using the model
        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], max_length=512, temperature=0.7, num_return_sequences=1)

        # Decode the generated output
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_code