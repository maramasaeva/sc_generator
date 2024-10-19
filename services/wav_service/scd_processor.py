# sc_generator/services/wav_service/scd_processor.py

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
        """
        Discover all .scd files in the scd_folder.
        """
        for filename in os.listdir(self.scd_folder):
            if filename.endswith('.scd'):
                scd_path = os.path.join(self.scd_folder, filename)
                scd_file = ScdFile(scd_path)
                scd_file.set_output_path(self.wav_folder)
                self.scd_files.append(scd_file)
        print(f"Discovered {len(self.scd_files)} .scd files.")

    def modify_and_save_scd_files(self):
        """
        Modify all discovered .scd files and save them in the scd_playable_folder.
        """
        for scd_file in self.scd_files:
            print(f"Modifying {scd_file.file_name}...")
            modified_file_path = self.converter.modify_scd_for_rendering(scd_file)
            playable_file_path = os.path.join(self.scd_playable_folder, os.path.basename(modified_file_path))
            os.rename(modified_file_path, playable_file_path)
            self.modified_files.append((playable_file_path, scd_file.output_path))
            print(f"Modified file saved as {playable_file_path}")

    def convert_to_wav(self):
        """
        Convert all modified .scd files in scd_playable_folder to .wav files in wav_folder.
        """
        for modified_file_path, output_path in self.modified_files:
            print(f"Converting {modified_file_path} to WAV...")
            self.converter.convert_to_wav(modified_file_path, output_path)
            print(f"Conversion of {modified_file_path} completed.")

    def process_all(self):
        """
        High-level method to process all .scd files: discover, modify, and convert.
        """
        self.discover_scd_files()
        self.modify_and_save_scd_files()
        self.convert_to_wav()