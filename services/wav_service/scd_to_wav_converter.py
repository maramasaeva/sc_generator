import os

class ScdToWavConverter:
    def __init__(self, server_options=None):
        self.server_options = server_options or {
            "numBuffers": 1024,
            "sampleRate": 44100,
            "numOutputBusChannels": 2
        }

    def modify_scd_for_rendering(self, scd_file):
        if scd_file.output_path is None:
            raise ValueError("Output path for the .wav file must be set before modification.")

        absolute_output_path = os.path.abspath(scd_file.output_path).replace("\\", "/")

        with open(scd_file.file_path, 'r') as file:
            scd_content = file.readlines()

        rendering_code = f"""
// NRT rendering setup
(
    var server, score, outputPath;
    outputPath = "{absolute_output_path}";

    server = Server(\\nrt, options: ServerOptions.new);
    server.options.numBuffers = {self.server_options['numBuffers']};
    server.options.sampleRate = {self.server_options['sampleRate']};
    server.options.numOutputBusChannels = {self.server_options['numOutputBusChannels']};

    score = Score([
        [0.0, [\\d_recv, SynthDef(\\exampleSynth, {{ Out.ar(0, SinOsc.ar(440) * 0.1) }}).asBytes]],  
        [0.1, [\\s_new, \\exampleSynth, 1000, 0, 0]],
        [5.0, [\\n_free, 1000]],
    ]);

    score.recordNRT(outputPath, "WAV", server.options.sampleRate, server.options.numOutputBusChannels, 
                    options: server.options, action: {{
        ("Rendered output: " ++ outputPath).postln;
        0.exit;
    }});
).fork;
        """

        # Add the rendering code at the end of the file
        modified_content = scd_content + [rendering_code]

        # Determine the modified file path (save in the same output directory but with '_modified' added to the name)
        modified_file_name = os.path.splitext(scd_file.file_name)[0] + "_modified.scd"
        modified_file_path = os.path.join(os.path.dirname(scd_file.output_path), modified_file_name)

        # Save the modified content as a new file
        with open(modified_file_path, 'w') as file:
            file.writelines(modified_content)

        print(f"Modified .scd file saved as {modified_file_path}")
        return modified_file_path

    # Comment out the convert_to_wav function as we are not using it right now.
    # def convert_to_wav(self, modified_file_path, output_path):
    #     # Command to run SuperCollider with the modified file
    #     command = ["/Applications/SuperCollider.app/Contents/MacOS/sclang", modified_file_path]
    #     os.system(' '.join(command))

    #     # Check if the file was created
    #     if os.path.isfile(output_path):
    #         print(f"Conversion of {modified_file_path} to {output_path} completed.")
    #     else:
    #         print(f"Error: Conversion failed for {modified_file_path}. No output file found at {output_path}.")