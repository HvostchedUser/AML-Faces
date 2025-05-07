# fiw_family_searcher/data_processing/embedder.py
from deepface import DeepFace
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import os

from fiw_family_searcher import config
from fiw_family_searcher.utils.helpers import setup_logger

logger = setup_logger(__name__)

# Initialize models globally to avoid reloading them repeatedly
# This can take time on first call
try:
    logger.info(f"Loading name embedding model: {config.NAME_EMBEDDING_MODEL}")
    name_model = SentenceTransformer(config.NAME_EMBEDDING_MODEL)
    logger.info("Name embedding model loaded.")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {e}")
    name_model = None


def get_face_embedding(image_path):
    """Generates face embedding using DeepFace."""
    if not os.path.exists(image_path):
        logger.warning(f"Image path does not exist: {image_path}")
        return None
    try:
        # DeepFace.represent returns a list of dictionaries, one for each detected face
        # Assuming one face per image or we take the first one
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name=config.FACE_EMBEDDING_MODEL_DEEPFACE,
            enforce_detection=False,  # Set to True if images might not be cropped faces
            detector_backend='skip'  # If images are already cropped faces
        )
        if embedding_objs and len(embedding_objs) > 0:
            # The embedding is under the 'embedding' key
            embedding = embedding_objs[0]['embedding']
            return np.array(embedding, dtype=np.float32)
        else:
            logger.warning(f"No face detected or embedding extracted for {image_path}")
            return None
    except Exception as e:
        # Common error: "Face could not be detected." if enforce_detection=True
        # or issues with image file itself.
        if "Face could not be detected" in str(e) or "No face detected" in str(e):
            logger.warning(f"No face detected in {image_path} by DeepFace. Error: {e}")
        else:
            logger.error(f"Error generating face embedding for {image_path}: {e}")
        return None


def get_name_embedding(text):
    """Generates name embedding using SentenceTransformers."""
    if name_model is None:
        logger.error("Name embedding model not loaded. Cannot generate name embedding.")
        return None
    if not text or not isinstance(text, str):
        logger.warning(f"Invalid text input for name embedding: {text}")
        return None
    try:
        embedding = name_model.encode(text)
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error generating name embedding for '{text}': {e}")
        return None


if __name__ == '__main__':
    # Test functions (create dummy image for testing)
    logger.info("Testing embedding functions...")

    # Create a dummy image file for testing get_face_embedding
    # Note: DeepFace needs a valid image with a face.
    # Using a placeholder path. Replace with an actual image path for real testing.
    dummy_image_path = "dummy_face.jpg"
    try:
        Image.new('RGB', (60, 30), color='red').save(dummy_image_path)
        logger.info(f"Created dummy image: {dummy_image_path}")
        # This will likely fail or return None if dummy_face.jpg is not a real face
        # face_emb = get_face_embedding(dummy_image_path)
        # if face_emb is not None:
        #     logger.info(f"Dummy Face embedding shape: {face_emb.shape}")
        # else:
        #     logger.warning("Could not get dummy face embedding (as expected for a non-face image).")
    except ImportError:
        logger.warning("Pillow is not installed. Cannot create dummy image for testing.")
    except Exception as e:
        logger.error(f"Error in dummy image creation/testing: {e}")
    finally:
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)

    name_emb = get_name_embedding("John Doe")
    if name_emb is not None:
        logger.info(f"Name embedding shape: {name_emb.shape}")
    else:
        logger.warning("Could not get name embedding.")

    logger.info("Embedding functions test complete.")