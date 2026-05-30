from app.core.utils import find_positions

import logging
logger = logging.getLogger("app.ingestion.chunks_generation.utils.processors")
logger.info("Loading file...")

import re

def join_texts(sentences: list[str]):
    if len(sentences) > 0:
        joined_texts = sentences[0] + "".join(" " + sentences[j+1] 
                                                    if sentences[j] and sentences[j][-1] != "\n" else sentences[j+1]
                                                    for j in range(len(sentences)-1))
    else:
        joined_texts = ""
    return joined_texts

def clean_transcription(transcription: str):
    """
    * Removes the repetition of section name followed by (continued)
    * Removes "ANNUAL REPORT 2025..." at the end of the pages
    * Removes the <CONT.> tag
    """
    # Continued sub-string removal
    continued_sub_string = "(continued)\n"
    len_continued_sub_string = len(continued_sub_string)
    while True:
        transcription_lower = transcription.lower()
        continued_sub_string_start = transcription_lower.find(continued_sub_string)
        if continued_sub_string_start == -1:
            break
        continued_sub_string_end = continued_sub_string_start + len_continued_sub_string
        start = find_positions(transcription[:continued_sub_string_end], pattern=re.compile(r"<H\d+>"))[-1]
        transcription = transcription[:start].strip() + '\n\n' + transcription[continued_sub_string_end:].strip()

    # ANNUAL REPORT 2025 REMOVAL
    pos_annual_report_2025 = transcription.find('ANNUAL REPORT 2025')
    if pos_annual_report_2025 > -1:
        transcription = transcription[:pos_annual_report_2025]

    # <CONT.> tag removal
    transcription = transcription.replace("<CONT.>", "")
    return transcription