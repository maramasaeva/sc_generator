import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model_and_tokenizer(model_name):
    print("Loading model and tokenizer...")
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def generate_modified_scd_code(model, tokenizer, prompt):
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output using the model
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=512, temperature=0.7, num_return_sequences=1)

    # Decode the generated output
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code


def process_scd_files(scd_folder, scd_playable_folder, model, tokenizer):
    os.makedirs(scd_playable_folder, exist_ok=True)

    for filename in os.listdir(scd_folder):
        if filename.endswith('.scd'):
            file_path = os.path.join(scd_folder, filename)

            # Read the content of the file
            with open(file_path, 'r') as file:
                original_code = file.read()

            # Prepare the prompt for the model
            prompt = f"""
            Original SuperCollider code:
            {original_code}

            Modify the code so that it plays and records the output to a file, saving the recording as a WAV file.
            """

            print(f"Processing file: {filename}")
            modified_code = generate_modified_scd_code(model, tokenizer, prompt)

            if modified_code:
                # Define the output path
                output_file_path = os.path.join(scd_playable_folder, filename)

                # Save the modified code
                with open(output_file_path, 'w') as file:
                    file.write(modified_code)

                print(f"Modified code saved to: {output_file_path}")
            else:
                print(f"Failed to process file: {filename}")


if __name__ == "__main__":
    # Define the paths
    scd_folder = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/scd"
    scd_playable_folder = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/scd_playable"

    # Use an open-source model like StarCoder or CodeGen
    model_name = "bigcode/starcoder"  # You can also use "Salesforce/codegen-350M-mono" or similar

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Process the SuperCollider files
    process_scd_files(scd_folder, scd_playable_folder, model, tokenizer)