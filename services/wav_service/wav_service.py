# wav_service.py

import os
from services.wav_service.scd_file import ScdFile
from services.wav_service.scd_to_wav_converter import ScdToWavConverter

class WavService:
    def __init__(self, scd_folder, output_folder):
        self.scd_folder = scd_folder
        self.output_folder = output_folder
        self.converter = ScdToWavConverter()
        self.scd_files = []
        self.modified_files = []

        # Ensure the output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def discover_scd_files(self):
        """
        Discover all .scd files in the given directory.
        """
        for filename in os.listdir(self.scd_folder):
            if filename.endswith('.scd'):
                scd_path = os.path.join(self.scd_folder, filename)
                scd_file = ScdFile(scd_path)
                scd_file.set_output_path(self.output_folder)
                self.scd_files.append(scd_file)
        print(f"Discovered {len(self.scd_files)} .scd files.")

    def modify_all(self):
        """
        Modify all discovered .scd files and save them separately.
        """
        for scd_file in self.scd_files:
            print(f"Modifying {scd_file.file_name}...")
            modified_file_path = self.converter.modify_scd_for_rendering(scd_file)
            self.modified_files.append((modified_file_path, scd_file.output_path))
            print(f"Modified file saved as {modified_file_path}")

    def convert_all(self):
        """
        Convert all modified .scd files to .wav files.
        """
        for modified_file_path, output_path in self.modified_files:
            print(f"Converting {modified_file_path} to WAV...")
            self.converter.convert_to_wav(modified_file_path, output_path)
            print(f"Conversion of {modified_file_path} completed.")