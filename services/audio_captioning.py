import os
from pathlib import Path
import torch
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from tqdm.auto import tqdm
from transformers import PreTrainedModel
from services.modelling_musilingo import MusilingoModel, LlamaForCausalLM, MusiLingo
from services.configuration_musilingo import MusiLingoConfig


def load_audio(audio_path, target_sr=24000, is_mono=True, is_normalize=False,
               crop_to_length_in_sample_points=None, crop_randomly=False, pad=False):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except RuntimeError as e:
        print(f"Error loading audio: {e}")
        raise RuntimeError("Failed to load the audio file. Please check the file format and dependencies.")

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    if is_mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if is_normalize:
        waveform = waveform / waveform.abs().max()

    if crop_to_length_in_sample_points:
        total_length = waveform.shape[1]
        desired_length = crop_to_length_in_sample_points
        if total_length > desired_length:
            start = torch.randint(0, total_length - desired_length + 1, (1,)).item() if crop_randomly else 0
            waveform = waveform[:, start:start + desired_length]
        elif pad:
            padding = desired_length - total_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
    return waveform


def get_musilingo_pred(model, tokenizer, text, audio_path, stopping, length_penalty=1, temperature=0.1,
                       max_new_tokens=300, num_beams=1, min_length=1, top_p=0.5, repetition_penalty=1.0):
    # Load and preprocess the audio
    audio = load_audio(audio_path, target_sr=24000, is_mono=True,
                       crop_to_length_in_sample_points=int(30 * 16000) + 1,
                       crop_randomly=True).cuda()

    # Use the processor associated with the model's configuration
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model.config.mert_model, trust_remote_code=True)
    audio_input = processor(audio, sampling_rate=24000, return_tensors="pt")['input_values'].cuda()

    # Encode audio using the already instantiated model
    audio_embeds, atts_audio = model.encode_audio(audio_input)

    # Prepare the prompt
    prompt = '<Audio><AudioHere></Audio> ' + text
    instruction_prompt = [model.prompt_template.format(prompt)]
    audio_embeds, atts_audio = model.instruction_prompt_wrap(audio_embeds, atts_audio, instruction_prompt)

    # Prepare inputs for generation
    tokenizer.padding_side = "right"
    batch_size = audio_embeds.shape[0]
    bos = torch.ones([batch_size, 1], dtype=torch.long, device='cuda') * tokenizer.bos_token_id
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
    if output_token[0] in [0, 1]:  # Remove unnecessary tokens
        output_token = output_token[1:]

    output_text = tokenizer.decode(output_token, add_special_tokens=False).split('###')[0]
    return output_text.split('Assistant:')[-1].strip()


if __name__ == '__main__':
    wav_directory = 'database/wav'
    captions_directory = 'database/captions'
    os.makedirs(captions_directory, exist_ok=True)

    config = MusiLingoConfig(
        mert_model="m-a-p/MERT-v1-330M",
        llama_model="m-a-p/MusiLingo-long-v1",
        prompt_template="<Audio><AudioHere></Audio>",
        max_txt_len=32,
        end_sym='\n'
    )

    # Instantiate the MusiLingo model
    musilingo = MusiLingoModel(config)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    musilingo.to("cuda")
    musilingo.eval()

    # Process WAV files
    process_wav_files(wav_directory, captions_directory, musilingo, tokenizer)