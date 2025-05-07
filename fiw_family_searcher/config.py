# fiw_family_searcher/config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(os.path.dirname(BASE_DIR), "public") # Assuming 'public' is one level up from project root

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
FACE_EMBEDDING_MODEL = "Facenet512" # Options: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"
NAME_EMBEDDING_MODEL = "all-MiniLM-L6-v2" # from sentence-transformers

# Embedding dimensions (Facenet512 is 512, all-MiniLM-L6-v2 is 384)
# DeepFace's represent function for Facenet512 returns a 128-dim embedding. Check docs or output.
# For "Facenet512" via DeepFace, it's usually 128. Let's verify or use a model with known dim.
# ArcFace often gives 512. Let's use ArcFace for higher dim.
FACE_EMBEDDING_MODEL_DEEPFACE = "ArcFace"
FACE_EMBEDDING_DIM = 512 # ArcFace default
NAME_EMBEDDING_DIM = 384 # all-MiniLM-L6-v2

# Search parameters
FAISS_SEARCH_TOP_K = 20 # How many results to fetch from FAISS initially
RRF_K = 60 # RRF constant for score calculation

# Classifier thresholds
FAMILY_MEMBERSHIP_THRESHOLD = 0.6 # Min probability to consider person belongs to family
STRONG_DIRECT_MATCH_SIMILARITY_THRESHOLD = 0.85 # For face similarity, if above this might be direct match

# Create directories if they don't exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Logging configuration (can be expanded in utils/helpers.py)
LOG_LEVEL = "INFO"