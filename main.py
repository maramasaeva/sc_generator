# main.py
import os
from scd_processor import ScdProcessor

if __name__ == "__main__":
    scd_folder = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/scd"
    scd_playable_folder = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/scd_playable"
    wav_folder = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/wav"

    processor = ScdProcessor(scd_folder, scd_playable_folder, wav_folder)
    processor.process_all()