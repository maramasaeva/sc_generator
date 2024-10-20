import os

class ScdFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.output_name = os.path.splitext(self.file_name)[0] + ".wav"
        self.output_path = None

    def set_output_path(self, output_folder):
        self.output_path = os.path.join(output_folder, self.output_name)

    def __repr__(self):
        return f"ScdFile(name={self.file_name}, output_path={self.output_path})"