# services/audio_captioning.py

import os
from pathlib import Path
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm.auto import tqdm

# Define the StoppingCriteriaSub class
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

# Define the load_audio function
def load_audio(audio_path, target_sr=24000, is_mono=True, is_normalize=False,
               crop_to_length_in_sample_points=None, crop_randomly=False, pad=False):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to target sample rate if necessary
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    # Convert to mono if necessary
    if is_mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Normalize waveform
    if is_normalize:
        waveform = waveform / waveform.abs().max()

    # Crop or pad waveform
    if crop_to_length_in_sample_points:
        total_length = waveform.shape[1]
        desired_length = crop_to_length_in_sample_points
        if total_length > desired_length:
            # Crop randomly or from the beginning
            if crop_randomly:
                start = torch.randint(0, total_length - desired_length + 1, (1,)).item()
            else:
                start = 0
            waveform = waveform[:, start:start + desired_length]
        elif pad:
            # Pad waveform to the desired length
            padding = desired_length - total_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
    return waveform

def get_musilingo_pred(model, tokenizer, text, audio_path, stopping, length_penalty=1, temperature=0.1,
                       max_new_tokens=300, num_beams=1, min_length=1, top_p=0.5, repetition_penalty=1.0):
    # Load and preprocess the audio
    audio = load_audio(audio_path, target_sr=24000,
                       is_mono=True,
                       is_normalize=False,
                       crop_to_length_in_sample_points=int(30*16000)+1,
                       crop_randomly=True,
                       pad=False).cuda()

    # Feature extraction
    processor = AutoProcessor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    audio_input = processor(audio, sampling_rate=24000, return_tensors="pt")['input_values'][0].cuda()

    # Encode audio using the model's custom method
    audio_embeds, atts_audio = model.encode_audio(audio_input)

    # Prepare the prompt
    prompt = '<Audio><AudioHere></Audio> ' + text
    instruction_prompt = [model.prompt_template.format(prompt)]
    audio_embeds, atts_audio = model.instruction_prompt_wrap(audio_embeds, atts_audio, instruction_prompt)

    # Prepare inputs for generation
    tokenizer.padding_side = "right"
    batch_size = audio_embeds.shape[0]
    bos = torch.ones([batch_size, 1], dtype=torch.long, device=torch.device('cuda')) * tokenizer.bos_token_id
    bos_embeds = model.llama_model.model.embed_tokens(bos)
    inputs_embeds = torch.cat([bos_embeds, audio_embeds], dim=1)

    # Generate the caption
    outputs = model.llama_model.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping,
        num_beams=num_beams,
        do_sample=True,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        temperature=temperature,
    )
    output_token = outputs[0]
    if output_token[0] == 0:  # Remove unknown token <unk> at the beginning
        output_token = output_token[1:]
    if output_token[0] == 1:  # Remove start token <s> at the beginning
        output_token = output_token[1:]
    output_text = tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # Remove stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    return output_text

def process_wav_files(wav_dir, captions_dir, model, tokenizer):
    wav_files = list(Path(wav_dir).glob('*.wav'))
    stopping = StoppingCriteriaList([
        StoppingCriteriaSub([
            torch.tensor([835]).cuda(),           # Adjust token IDs as necessary
            torch.tensor([2277, 29937]).cuda()
        ])
    ])
    for wav_file in tqdm(wav_files, desc='Processing audio files'):
        caption = get_musilingo_pred(
            model, tokenizer, "Please provide a detailed description of this music.", str(wav_file), stopping
        )
        caption_file = Path(captions_dir) / (wav_file.stem + '.txt')
        with open(caption_file, 'w') as f:
            f.write(caption)
        print(f"Caption for {wav_file.name}: {caption}")

if __name__ == '__main__':
    # Paths to your audio files and captions directory
    wav_directory = 'database/wav'
    captions_directory = 'database/captions'
    os.makedirs(captions_directory, exist_ok=True)

    # Load the MusiLingo model and tokenizer
    musilingo = AutoModel.from_pretrained("m-a-p/MusiLingo-long-v1", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("m-a-p/MusiLingo-long-v1", trust_remote_code=True)
    musilingo.to("cuda")
    musilingo.eval()

    # Process the audio files
    process_wav_files(wav_directory, captions_directory, musilingo, tokenizer)