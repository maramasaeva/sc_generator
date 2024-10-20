# main.py

import os
from services.audio_captioning import process_wav_files
from services.generate_music import analyze_poem, find_matching_captions, compute_similarity
from services.generate_scd import generate_supercollider_code

def main():
    # Step 1: Generate captions for audio files
    print("Generating captions for audio files...")
    if not os.path.exists('database/captions'):
        os.makedirs('database/captions')
    process_wav_files('database/wav', 'database/captions')

    # Step 2: Get user input
    poem = input("Enter your poem:\n")
    poem_features = analyze_poem(poem)

    # Step 3: Find matching caption
    matching_caption = find_matching_captions(poem_features, 'database/captions')
    if matching_caption is None:
        print("No matching captions found.")
        return

    # Step 4: Generate SuperCollider code
    prompt = (
        f"Poem:\n{poem}\n\n"
        f"Musical Description:\n{matching_caption}\n\n"
        "Using the emotions and themes from the poem and the musical description, "
        "generate SuperCollider code that evokes these emotions and themes."
    )
    scd_code = generate_supercollider_code(prompt)
    with open('output.scd', 'w') as f:
        f.write(scd_code)
    print("Generated SuperCollider code saved to output.scd")

if __name__ == '__main__':
    main()