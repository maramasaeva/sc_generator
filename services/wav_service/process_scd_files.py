import os
import time
import subprocess


class SupercolliderProcessor:
    def __init__(self, scd_folder, wav_folder):
        self.scd_folder = scd_folder
        self.wav_folder = wav_folder

    def process_files(self):
        for filename in os.listdir(self.scd_folder):
            if filename.endswith(".scd"):
                file_path = os.path.join(self.scd_folder, filename)
                print(f"Processing {filename}...")
                self._process_file(file_path, filename)

    def _process_file(self, file_path, original_filename):
        # Start the SuperCollider server
        print("Starting SuperCollider server...")
        subprocess.run(["sclang", file_path], check=True)

        # Wait for the server to boot
        time.sleep(5)

        # Path where SuperCollider saves recordings
        sc_recording_path = "/Users/maramasaeva/Music/SuperCollider Recordings/SC_Recording.wav"

        # Wait for the recording to complete
        print("Waiting for 10 seconds while recording...")
        time.sleep(10)

        # Check if the recording exists
        if os.path.exists(sc_recording_path):
            # Destination path for the recorded file
            output_file = os.path.join(self.wav_folder, f"{os.path.splitext(original_filename)[0]}.wav")
            print(f"Recording complete. Saving file to {output_file}...")

            # Copy the recorded file to the destination
            shutil.copy(sc_recording_path, output_file)
        else:
            print(f"Error: Could not find recording file at {sc_recording_path}.")