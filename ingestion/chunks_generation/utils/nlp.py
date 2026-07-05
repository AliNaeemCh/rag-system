import logging
logger = logging.getLogger("ingestion.chunks_generation.utils.nlp")
logger.info("Loading file...")

import spacy

nlp = spacy.load("en_core_web_sm")