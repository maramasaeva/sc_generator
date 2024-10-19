(
var outputPath = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/wav_files/sc_test.wav";  // Set the output file path
var buf, synth;

s.waitForBoot({
    // Allocate a buffer for recording (60 seconds of stereo audio)
    buf = Buffer.alloc(s, s.sampleRate * 60.0, 2);

    // Define a SynthDef for playback and recording
    SynthDef(\playAndRecord, { |out=0, bufnum|
        var signal = SinOsc.ar(OnePole.ar(Mix(LFSaw.ar([1, 0.99], [0, 0.6], 2000, 2000).trunc([400, 600]) * [1, -1]), 0.98)).dup * 0.1;
        Out.ar(out, signal);
        DiskOut.ar(bufnum, signal);  // Record the output signal to the buffer
    }).add;

    s.sync;

    // Play and record the synth
    synth = Synth(\playAndRecord, [\bufnum, buf]);

    // Schedule stopping and saving the recording after 10 seconds
    SystemClock.sched(10, {
        synth.free;  // Free the synth
        buf.write(outputPath, "WAV", "int16", completionMessage: {
            buf.free;
            ("Recording saved to: " ++ outputPath).postln;
        });
    });
});
)