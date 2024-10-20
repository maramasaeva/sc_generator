# services/generate_music.py

import spacy
from pathlib import Path
from services.generate_scd import generate_supercollider_code
import os
import sys

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def analyze_poem(poem):
    doc = nlp(poem)
    # Extract nouns, verbs, and adjectives as keywords
    keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'VERB', 'ADJ')]
    poem_features = {
        'keywords': keywords,
    }
    return poem_features

def find_matching_captions(poem_features, captions_dir):
    captions_files = Path(captions_dir).glob('*.txt')
    best_match = None
    highest_score = 0
    for caption_file in captions_files:
        with open(caption_file, 'r') as f:
            caption = f.read()
        score = compute_similarity(poem_features, caption)
        if score > highest_score:
            highest_score = score
            best_match = caption
    return best_match

def compute_similarity(poem_features, caption):
    # Simple keyword overlap
    caption_doc = nlp(caption)
    caption_keywords = [token.lemma_ for token in caption_doc if token.pos_ in ('NOUN', 'VERB', 'ADJ')]
    overlap = set(poem_features['keywords']).intersection(set(caption_keywords))
    return len(overlap)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python generate_music.py 'Your poem here'")
        sys.exit(1)

    poem = sys.argv[1]
    poem_features = analyze_poem(poem)
    captions_directory = 'database/captions'
    if not os.path.exists(captions_directory):
        print("Captions directory not found. Please run audio_captioning.py first.")
        sys.exit(1)
    matching_caption = find_matching_captions(poem_features, captions_directory)
    if matching_caption is None:
        print("No matching captions found.")
        sys.exit(1)

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