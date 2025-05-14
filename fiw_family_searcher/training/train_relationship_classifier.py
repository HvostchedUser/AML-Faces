import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from tqdm import tqdm
import os

from fiw_family_searcher import config
from fiw_family_searcher.utils.helpers import setup_logger, load_pickle, save_pickle
from fiw_family_searcher.data_processing.embedder import get_face_embedding  # For ad-hoc if needed

logger = setup_logger(__name__)


def generate_training_features_relationship():
    logger.info("Generating training features for relationship classifier...")
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    # Load pre-computed average face embeddings (or first photo embedding as representative)
    avg_face_embs_path = os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl")
    if not os.path.exists(avg_face_embs_path):
        logger.error(f"Average face embeddings file not found: {avg_face_embs_path}. "
                     "Run family classifier training or a dedicated script to create it.")
        # Fallback: try to generate on the fly (very slow, from train_family_classifier.py logic)
        # This is a dependency. For simplicity, assume it exists.
        return None, None

    person_avg_face_embeddings = load_pickle(avg_face_embs_path)
    logger.info(f"Loaded {len(person_avg_face_embeddings)} pre-computed average face embeddings.")

    # Get all relationships from the database
    # Person1ID, Person2ID, RID
    cursor.execute("SELECT Person1ID, Person2ID, RID FROM Relationships WHERE RID != 0")  # Exclude self/NA
    relationship_pairs = cursor.fetchall()

    logger.info(f"Found {len(relationship_pairs)} relationship pairs for training.")

    features_list = []
    labels_list = []  # RID will be the label (multi-class)

    for p1_id, p2_id, rid in tqdm(relationship_pairs, desc="Generating relationship features"):
        p1_face_emb = person_avg_face_embeddings.get(p1_id)
        p2_face_emb = person_avg_face_embeddings.get(p2_id)

        if p1_face_emb is None or p2_face_emb is None:
            logger.debug(f"Skipping pair ({p1_id}, {p2_id}) due to missing face embedding(s).")
            continue

        # Feature ideas:
        # 1. Concatenation: [p1_face_emb, p2_face_emb]
        # 2. Difference: abs(p1_face_emb - p2_face_emb)
        # 3. Element-wise product: p1_face_emb * p2_face_emb
        # Combining these can be effective.

        # Let's use concatenation as a starting point, it's simple and CatBoost can find interactions.
        # feature_vector = np.concatenate((p1_face_emb, p2_face_emb))

        # Or, use all three:
        diff_emb = np.abs(p1_face_emb - p2_face_emb)
        prod_emb = p1_face_emb * p2_face_emb
        feature_vector = np.concatenate((p1_face_emb, p2_face_emb, diff_emb, prod_emb))

        features_list.append(feature_vector)
        labels_list.append(rid)  # RID is the target class

    conn.close()
    if not features_list:
        logger.error("No features were generated for relationship classifier. Aborting training.")
        return None, None

    return np.array(features_list), np.array(labels_list)


def train_model():
    X, y = generate_training_features_relationship()

    if X is None or y is None or len(X) == 0:
        logger.error("Feature generation failed or yielded no data. Classifier training aborted.")
        return

    logger.info(f"Generated {len(X)} samples for relationship training. Feature vector length: {X.shape[1]}")
    unique_labels, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique_labels, counts))}")

    # Check if there are enough classes / samples per class for CatBoost
    if len(unique_labels) < 2:
        logger.error(f"Not enough classes ({len(unique_labels)}) for multi-class classification. Need at least 2.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = CatBoostClassifier(
        iterations=700,  # Potentially more iterations for complex embedding features
        learning_rate=0.03,
        depth=7,  # Potentially deeper for embedding features
        loss_function='MultiClass',
        eval_metric='Accuracy',  # Or MultiClass Logloss
        random_seed=42,
        verbose=100,
        # task_type="GPU" # If GPU is available and faiss-gpu/catboost-gpu installed
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

    logger.info("Relationship classifier training complete.")
    model.save_model(config.RELATIONSHIP_CLASSIFIER_PATH)
    logger.info(f"Relationship classifier saved to {config.RELATIONSHIP_CLASSIFIER_PATH}")

    # Optional: Print feature importances (can be very long for embedding features)
    # try:
    #     importances = model.get_feature_importance()
    #     logger.info(f"Top 10 Feature Importances: {importances[:10]}")
    # except Exception as e:
    #     logger.warning(f"Could not get/print feature importances: {e}")


if __name__ == '__main__':
    # This script depends on "person_avg_face_embeddings.pkl"
    # Ensure it's created by running train_family_classifier.py first, or a dedicated script.
    if not os.path.exists(os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl")):
        logger.error("person_avg_face_embeddings.pkl not found. "
                     "Please run family classifier training first, which might create it, "
                     "or ensure a dedicated script generates this file.")
    else:
        train_model()