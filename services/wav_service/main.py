import os
from services.wav_service.scd_processor import ScdProcessor
from services.llm_service.llm_processor import LLMProcessor

if __name__ == "__main__":
    scd_folder = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/scd"
    scd_playable_folder = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/scd_playable"
    wav_folder = "/Users/maramasaeva/Documents/SC/SC_GENERATOR/sc_generator/database/wav"

    processor = ScdProcessor(scd_folder, scd_playable_folder, wav_folder)
    processor.process_all()

    llm_processor = LLMProcessor()
    llm_processor.modify_scd_files(scd_folder, scd_playable_folder)