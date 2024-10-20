import torch
import os
from pathlib import Path
from services.audio_captioning import get_musilingo_pred, load_audio
from services.configuration_musilingo import MusiLingoConfig, PATH
from services.modelling_musilingo import MusilingoModel

from transformers import AutoTokenizer, StoppingCriteriaList, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, Wav2Vec2FeatureExtractor


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


if __name__ == '__main__':
    # Paths to your audio files and captions directory
    wav_directory = 'test/database/wav'
    captions_directory = 'test/database/captions'
    os.makedirs(captions_directory, exist_ok=True)

    # Load the MusiLingo model and processor
    mert_model_name = "m-a-p/MERT-v1-330M"
    musilingo_model_name = "m-a-p/MusiLingo-long-v1"

    # Load the processor (feature extractor) for the audio
    processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_model_name, trust_remote_code=True)

    # Load the MusiLingo model
    musilingo = AutoModel.from_pretrained(musilingo_model_name, trust_remote_code=True)
    musilingo.to("cuda")
    musilingo.eval()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # Prepare the stopping criteria
    stopping = StoppingCriteriaList([
        StoppingCriteriaSub([
            torch.tensor([835]).cuda(),  # Adjust token IDs as necessary
            torch.tensor([2277, 29937]).cuda()
        ])
    ])

    # Iterate over the audio files in the directory
    for wav_file in Path(wav_directory).glob('*.wav'):
        print(f"Processing {wav_file.name}")
        caption = get_musilingo_pred(
            model=musilingo,
            tokenizer=tokenizer,
            text="Please provide a detailed description of this music.",
            audio_path=str(wav_file),
            stopping=stopping
        )
        caption_file = Path(captions_directory) / (wav_file.stem + '.txt')
        with open(caption_file, 'w') as f:
            f.write(caption)
        print(f"Caption for {wav_file.name}: {caption}")