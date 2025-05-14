# fiw_family_searcher/config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR) # aml-wild-face-detector/
PUBLIC_DIR = os.path.join(PROJECT_ROOT_DIR, "public") # Assuming 'public' is one level up from project root

# Data paths
FIW_RIDS_CSV = os.path.join(PUBLIC_DIR, "FIW_RIDs.csv")
FIW_FIDS_CSV = os.path.join(PUBLIC_DIR, "FIW_FIDs.csv")
FIDS_FULL_DIR = os.path.join(PUBLIC_DIR, "FIDs_full", "FIDs")

# Database
DB_PATH = os.path.join(BASE_DIR, "fiw_database.sqlite")
DB_CONN_STRING = f"sqlite:///{DB_PATH}"

# FAISS Index paths
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
FACE_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "face_index.faiss")
NAME_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "name_index.faiss")
FACE_ID_MAP_PATH = os.path.join(EMBEDDINGS_DIR, "face_id_map.pkl")
NAME_ID_MAP_PATH = os.path.join(EMBEDDINGS_DIR, "name_id_map.pkl")

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
FAMILY_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "family_membership_classifier.cbm")
RELATIONSHIP_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "relationship_classifier.cbm")

# Embedding models
FACE_EMBEDDING_MODEL_DEEPFACE = "ArcFace" # Ensure this model works well with SHAP GradientExplainer (TF-based)
FACE_EMBEDDING_DIM = 512 # ArcFace default
NAME_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NAME_EMBEDDING_DIM = 384

# Search parameters
FAISS_SEARCH_TOP_K = 20
RRF_K = 60

# Classifier thresholds
FAMILY_MEMBERSHIP_THRESHOLD = 0.6
STRONG_DIRECT_MATCH_SIMILARITY_THRESHOLD = 0.85

# Explanation settings
EXPLAINABILITY_DIR_NAME = "explanations" # Subdirectory in interface/static for explanation images
EXPLANATIONS_DIR_ABS_PATH = os.path.join(PROJECT_ROOT_DIR, "interface", "static", EXPLAINABILITY_DIR_NAME)
EXPLANATIONS_URL_PREFIX = f"/static/{EXPLAINABILITY_DIR_NAME}" # URL prefix to serve these images

# Create directories if they don't exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPLANATIONS_DIR_ABS_PATH, exist_ok=True)


# Logging configuration
LOG_LEVEL = "INFO"