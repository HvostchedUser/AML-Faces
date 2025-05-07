# fiw_family_searcher/utils/helpers.py
import logging
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    # Ensure embeddings are 2D arrays for cosine_similarity
    emb1_2d = np.array(emb1).reshape(1, -1)
    emb2_2d = np.array(emb2).reshape(1, -1)
    return cosine_similarity(emb1_2d, emb2_2d)[0][0]
