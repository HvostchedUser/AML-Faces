# fiw_family_searcher/search/faiss_search.py
import faiss
import numpy as np
import os

from fiw_family_searcher import config
from fiw_family_searcher.utils.helpers import load_pickle, setup_logger

logger = setup_logger(__name__)


class FaissSearcher:
    def __init__(self):
        self.face_index = None
        self.name_index = None
        self.face_id_map = None
        self.name_id_map = None
        self._load_indexes()

    def _load_indexes(self):
        if os.path.exists(config.FACE_INDEX_PATH) and os.path.exists(config.FACE_ID_MAP_PATH):
            try:
                self.face_index = faiss.read_index(config.FACE_INDEX_PATH)
                self.face_id_map = load_pickle(config.FACE_ID_MAP_PATH)
                logger.info(f"Face FAISS index loaded with {self.face_index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Error loading face FAISS index or map: {e}")
        else:
            logger.warning("Face FAISS index or ID map not found. Face search will not be available.")

        if os.path.exists(config.NAME_INDEX_PATH) and os.path.exists(config.NAME_ID_MAP_PATH):
            try:
                self.name_index = faiss.read_index(config.NAME_INDEX_PATH)
                self.name_id_map = load_pickle(config.NAME_ID_MAP_PATH)
                logger.info(f"Name FAISS index loaded with {self.name_index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Error loading name FAISS index or map: {e}")
        else:
            logger.warning("Name FAISS index or ID map not found. Name search will not be available.")

    def search_face(self, query_embedding: np.ndarray, k: int = config.FAISS_SEARCH_TOP_K):
        """Searches the face index."""
        if self.face_index is None or self.face_id_map is None:
            logger.warning("Face index not loaded. Cannot perform face search.")
            return [], []
        if query_embedding is None:
            logger.warning("Query face embedding is None. Cannot search.")
            return [], []

        query_embedding = np.array([query_embedding]).astype('float32')
        # Ensure query embedding dim matches index dim
        if query_embedding.shape[1] != self.face_index.d:
            logger.error(f"Query face embedding dimension ({query_embedding.shape[1]}) "
                         f"does not match FAISS index dimension ({self.face_index.d}).")
            return [], []

        distances, indices = self.face_index.search(query_embedding, k)

        results_person_ids = [self.face_id_map[i] for i in indices[0] if i != -1]  # i == -1 if fewer than k results
        results_distances = [d for i, d in enumerate(distances[0]) if indices[0][i] != -1]

        return results_person_ids, results_distances

    def search_name(self, query_embedding: np.ndarray, k: int = config.FAISS_SEARCH_TOP_K):
        """Searches the name index."""
        if self.name_index is None or self.name_id_map is None:
            logger.warning("Name index not loaded. Cannot perform name search.")
            return [], []
        if query_embedding is None:
            logger.warning("Query name embedding is None. Cannot search.")
            return [], []

        query_embedding = np.array([query_embedding]).astype('float32')
        if query_embedding.shape[1] != self.name_index.d:
            logger.error(f"Query name embedding dimension ({query_embedding.shape[1]}) "
                         f"does not match FAISS index dimension ({self.name_index.d}).")
            return [], []

        distances, indices = self.name_index.search(query_embedding, k)

        results_person_ids = [self.name_id_map[i] for i in indices[0] if i != -1]
        results_distances = [d for i, d in enumerate(distances[0]) if indices[0][i] != -1]

        return results_person_ids, results_distances

# Global instance
# faiss_searcher_instance = FaissSearcher() # Instantiate when needed by QueryProcessor