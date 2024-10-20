import os
from services.llm_service.llm_service import LLMService


class LLMProcessor:
    def __init__(self):
        self.llm_service = LLMService()

    def modify_scd_files(self, scd_folder, scd_playable_folder):
        os.makedirs(scd_playable_folder, exist_ok=True)

        for filename in os.listdir(scd_folder):
            if filename.endswith('.scd'):
                file_path = os.path.join(scd_folder, filename)

                with open(file_path, 'r') as file:
                    original_code = file.read()

                prompt = f"""
                Original SuperCollider code:
                {original_code}

                Modify the Supercollider code so that, upon selecting all of the code and running it, the complete audio will play in one run.
                
                
                """

                print(f"Processing file: {filename}")
                modified_code = self.llm_service.generate_modified_scd_code(prompt)

                if modified_code:
                    output_file_path = os.path.join(scd_playable_folder, filename)
                    with open(output_file_path, 'w') as file:
                        file.write(modified_code)
                    print(f"Modified code saved to: {output_file_path}")
                else:
                    print(f"Failed to process file: {filename}")