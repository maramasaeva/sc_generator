import os
import unittest
from services.wav_service.scd_to_wav_converter import ScdToWavConverter
from services.wav_service.scd_file import ScdFile


class TestScdToWavConversion(unittest.TestCase):
    def setUp(self):
        """
        Set up the paths for the test.
        """
        # Paths to the scd file and the output directory
        self.scd_folder = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/scd_files"
        self.output_folder = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/wav_files"
        self.scd_file_path = os.path.join(self.scd_folder, "sc_test.scd")
        self.scd_file = ScdFile(self.scd_file_path)

        # Set the output path for the wav file
        self.scd_file.set_output_path(self.output_folder)

        # Instantiate the converter
        self.converter = ScdToWavConverter()

    def test_modify_scd_for_rendering(self):
        """
        Test if the .scd file is modified and saved correctly.
        """
        modified_file_path = self.converter.modify_scd_for_rendering(self.scd_file)
        self.assertTrue(os.path.isfile(modified_file_path), "The modified .scd file was not created.")
        print(f"Modified .scd file created at: {modified_file_path}")

    def test_convert_to_wav(self):
        """
        Test the conversion from the modified .scd file to a .wav file.
        """
        # First, modify the scd file
        modified_file_path = self.converter.modify_scd_for_rendering(self.scd_file)

        # Convert the modified scd file to wav
        self.converter.convert_to_wav(modified_file_path, self.scd_file.output_path)

        # Check if the .wav file was created
        self.assertTrue(os.path.isfile(self.scd_file.output_path), "The .wav file was not created.")
        print(f"Conversion successful: .wav file created at: {self.scd_file.output_path}")

    def tearDown(self):
        """
        Clean up any generated files after tests.
        """
        # Delete the modified .scd file if it exists
        modified_file_name = os.path.splitext(self.scd_file.file_name)[0] + "_modified.scd"
        modified_file_path = os.path.join(self.output_folder, modified_file_name)
        if os.path.isfile(modified_file_path):
            os.remove(modified_file_path)
            print(f"Removed modified .scd file: {modified_file_path}")

        # Delete the .wav file if it exists
        if os.path.isfile(self.scd_file.output_path):
            os.remove(self.scd_file.output_path)
            print(f"Removed generated .wav file: {self.scd_file.output_path}")


if __name__ == "__main__":
    unittest.main()