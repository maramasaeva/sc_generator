# scd_processor.py

import os
from services.wav_service.scd_file import ScdFile
from services.wav_service.scd_to_wav_converter import ScdToWavConverter

class ScdProcessor:
    def __init__(self, scd_folder, scd_playable_folder, wav_folder):
        self.scd_folder = scd_folder
        self.scd_playable_folder = scd_playable_folder
        self.wav_folder = wav_folder
        self.converter = ScdToWavConverter()
        self.scd_files = []
        self.modified_files = []

        # Ensure output folders exist
        os.makedirs(self.scd_playable_folder, exist_ok=True)
        os.makedirs(self.wav_folder, exist_ok=True)

    def discover_scd_files(self):
        for filename in os.listdir(self.scd_folder):
            if filename.endswith('.scd'):
                scd_path = os.path.join(self.scd_folder, filename)
                scd_file = ScdFile(scd_path)
                scd_file.set_output_path(self.wav_folder)
                self.scd_files.append(scd_file)
        print(f"Discovered {len(self.scd_files)} .scd files.")

    def modify_and_save_scd_files(self):
        for scd_file in self.scd_files:
            print(f"Modifying {scd_file.file_name}...")
            modified_file_path = self.converter.modify_scd_for_rendering(scd_file)
            playable_file_path = os.path.join(self.scd_playable_folder, os.path.basename(modified_file_path))
            os.rename(modified_file_path, playable_file_path)
            self.modified_files.append((playable_file_path, scd_file.output_path))
            print(f"Modified file saved as {playable_file_path}")

    def process_all(self):
        self.discover_scd_files()
        self.modify_and_save_scd_files()