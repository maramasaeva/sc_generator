# test/test_audio_captioning.py

import os
import sys
from pathlib import Path
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from services.audio_captioning import (
    load_audio,
    get_musilingo_pred,
    StoppingCriteriaSub
)
from transformers import StoppingCriteriaList

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_audio_captioning.py path/to/audio_file.wav")
        sys.exit(1)

    # Get the audio file path from command-line arguments
    audio_file = sys.argv[1]
    captions_directory = os.path.join(os.path.dirname(audio_file), '..', 'captions')
    os.makedirs(captions_directory, exist_ok=True)

    # Initialize the model and tokenizer
    musilingo = AutoModel.from_pretrained("m-a-p/MusiLingo-long-v1", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("m-a-p/MusiLingo-long-v1", trust_remote_code=True)
    musilingo.to("cuda")
    musilingo.eval()

    # Check if the audio file exists
    if not os.path.exists(audio_file):
        print(f"Audio file {audio_file} does not exist.")
        sys.exit(1)

    # Define the stopping criteria
    stopping = StoppingCriteriaList([
        StoppingCriteriaSub([
            torch.tensor([835]).cuda(),           # Adjust token IDs as necessary
            torch.tensor([2277, 29937]).cuda()
        ])
    ])

    # Generate caption for the audio file
    print(f"Processing {audio_file}")
    caption = get_musilingo_pred(
        musilingo, tokenizer, "Please provide a detailed description of this music.", audio_file, stopping
    )

    # Save the caption
    audio_filename = Path(audio_file).stem
    caption_file = Path(captions_directory) / f'{audio_filename}.txt'
    with open(caption_file, 'w') as f:
        f.write(caption)
    print(f"Caption saved to {caption_file}")
    print(f"Caption: {caption}")