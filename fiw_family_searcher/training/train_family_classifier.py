# fiw_family_searcher/training/train_family_classifier.py
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from tqdm import tqdm
import random

from fiw_family_searcher import config
from fiw_family_searcher.utils.helpers import setup_logger, save_pickle, load_pickle, calculate_cosine_similarity
from fiw_family_searcher.data_processing.embedder import get_face_embedding, \
    get_name_embedding  # For ad-hoc cases if needed

logger = setup_logger(__name__)


def get_person_embeddings_from_db(conn, person_id):
    """Helper to fetch stored embeddings for a person."""
    cursor = conn.cursor()
    # Name embedding
    cursor.execute("SELECT NameEmbedding FROM Persons WHERE PersonID = ?", (person_id,))
    row = cursor.fetchone()
    name_emb = np.frombuffer(row[0], dtype=np.float32) if row and row[0] else None

    # Face embeddings (average of all photos for this person)
    cursor.execute("""
        SELECT Photos.PhotoPath FROM Photos 
        WHERE Photos.PersonID = ?
    """, (person_id,))
    # This is inefficient if done repeatedly. Pre-calculate average face embeddings.
    # For now, we'll use an ad-hoc one if needed, or better, use stored embeddings directly.
    # The features should use the *input* person's embedding and compare against *family members*.
    # For training, the "input person" is one of the DB persons.

    # Fetch *one* representative face embedding for this person.
    # For simplicity, this requires re-computing or having a pre-computed representative.
    # Let's assume we can get a primary face embedding for each person.
    # For now, we'll load it from a pre-generated list if available, or compute one.
    # This part needs careful thought for efficiency during feature generation.
    # Let's make a placeholder for getting a single face embedding for a person.
    # A better way: pre-calculate average face embeddings per person and store them.
    # For this script, we will take the first photo's embedding.

    first_photo_path_row = cursor.execute("SELECT PhotoPath FROM Photos WHERE PersonID = ? LIMIT 1",
                                          (person_id,)).fetchone()
    face_emb = None
    if first_photo_path_row:
        # This is slow if done for all persons during feature generation.
        # face_emb = get_face_embedding(first_photo_path_row[0])
        # Instead, we should have these pre-computed and stored or loadable.
        # Let's simulate this by trying to get it from a pre-built FAISS-like structure if available
        # Or, for training, we might need to re-embed. This is a bottleneck.

        # Hack: for training, we need embeddings. Database stores name_emb. Face_embs are in FAISS.
        # We need a way to get a representative face_emb for PersonID
        # This should be prepared once.
        # For now, let's assume `all_face_embeddings` dict {PersonID: [emb1, emb2...]} is pre-loaded.
        # For simplicity in this script: query FAISS for the person's own embedding (not ideal)
        # Or, best: iterate through FAISS map, get embeddings.
        pass  # This function signature needs rethinking based on available precomputed data.

    return face_emb, name_emb


def generate_training_features_family(num_negative_samples_per_person=3):
    logger.info("Generating training features for family membership classifier...")
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    # Load all person data and their pre-computed embeddings
    # Persons: PersonID, FID, NameEmbedding
    # We also need representative face embeddings for each person.
    # Option 1: Load from FAISS (inefficient to query one by one)
    # Option 2: Re-embed first photo (slow)
    # Option 3: Pre-calculate and store avg face embeddings per person in a new table or pkl file. This is best.

    # Let's assume we have a helper that gives us avg_face_emb and name_emb for a PersonID
    # For now, create a temporary dict of embeddings by loading them.

    logger.info("Loading pre-computed embeddings for all persons...")
    all_persons_details = {}  # {PersonID: {'name_emb': emb, 'avg_face_emb': emb, 'FID': fid}}

    # Load name embeddings
    cursor.execute("SELECT PersonID, NameEmbedding, FID FROM Persons")
    for pid, name_emb_blob, fid_val in tqdm(cursor.fetchall(), desc="Loading name embeddings"):
        if name_emb_blob:
            all_persons_details[pid] = {
                'name_emb': np.frombuffer(name_emb_blob, dtype=np.float32),
                'avg_face_emb': None,  # To be filled
                'fid': fid_val
            }
        else:  # Should not happen if DB setup is complete
            all_persons_details[pid] = {'name_emb': None, 'avg_face_emb': None, 'fid': fid_val}

    # Load or compute average face embeddings
    # This is a critical step for efficiency. We'll simulate having them.
    # A separate script should ideally compute and save these.
    # E.g., `person_avg_face_embeddings.pkl` = {PersonID: avg_face_embedding_vector}
    # For this example, let's try to get first photo embedding for each person. This WILL BE SLOW.
    # To make this runnable, let's use a simplified approach or mock it.
    # REALISTIC APPROACH: Create a pickled file of {PersonID: avg_face_emb} during setup.

    # For this script, let's assume `person_avg_face_embeddings.pkl` was created by `database_setup.py` or another script.
    # If not, this part will be slow.
    avg_face_embs_path = os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl")
    if os.path.exists(avg_face_embs_path):
        logger.info(f"Loading pre-computed average face embeddings from {avg_face_embs_path}")
        person_avg_face_embeddings = load_pickle(avg_face_embs_path)
        for pid, emb in person_avg_face_embeddings.items():
            if pid in all_persons_details:
                all_persons_details[pid]['avg_face_emb'] = emb
    else:
        logger.warning(f"{avg_face_embs_path} not found. Computing on the fly (SLOW). Create this file for efficiency.")
        temp_avg_face_embeddings = {}
        cursor.execute("SELECT PersonID, PhotoPath FROM Photos GROUP BY PersonID")  # Get one photo per person
        for pid, photo_path in tqdm(cursor.fetchall(), desc="Computing placeholder face embeddings"):
            if pid in all_persons_details:
                emb = embedder.get_face_embedding(photo_path)  # Using first photo as representative
                all_persons_details[pid]['avg_face_emb'] = emb
                temp_avg_face_embeddings[pid] = emb
        save_pickle(temp_avg_face_embeddings, avg_face_embs_path)  # Save for next time
        logger.info(f"Saved placeholder average face embeddings to {avg_face_embs_path}")

    # Get all FIDs and their members
    families_members = {}  # {FID: [PersonID1, PersonID2,...]}
    for pid, data in all_persons_details.items():
        fid = data['fid']
        if fid not in families_members:
            families_members[fid] = []
        families_members[fid].append(pid)

    all_fids = list(families_members.keys())
    features_list = []
    labels_list = []

    logger.info("Generating positive and negative samples...")
    for person_id, p_details in tqdm(all_persons_details.items(), desc="Generating training samples"):
        input_person_face_emb = p_details['avg_face_emb']
        input_person_name_emb = p_details['name_emb']
        actual_fid = p_details['fid']

        if input_person_face_emb is None:  # Skip if no face embedding for the input person
            continue

        # Positive Sample
        target_family_members = [m for m in families_members[actual_fid] if m != person_id]
        if not target_family_members:  # Single person family or person is the only one with embedding
            continue

        feat_vector_pos = _extract_family_features(
            input_person_face_emb, input_person_name_emb,
            target_family_members, all_persons_details
        )
        if feat_vector_pos:
            features_list.append(feat_vector_pos)
            labels_list.append(1)

        # Negative Samples
        neg_samples_count = 0
        attempts = 0
        max_attempts = num_negative_samples_per_person * 5  # Try a few times to find different families

        while neg_samples_count < num_negative_samples_per_person and attempts < max_attempts:
            attempts += 1
            candidate_neg_fid = random.choice(all_fids)
            if candidate_neg_fid == actual_fid:
                continue

            negative_family_members = families_members[candidate_neg_fid]  # All members, including potential matches

            feat_vector_neg = _extract_family_features(
                input_person_face_emb, input_person_name_emb,
                negative_family_members, all_persons_details
            )
            if feat_vector_neg:
                features_list.append(feat_vector_neg)
                labels_list.append(0)
                neg_samples_count += 1

    conn.close()
    if not features_list:
        logger.error("No features were generated. Aborting training.")
        return None, None

    return np.array(features_list), np.array(labels_list)


def _extract_family_features(input_face_emb, input_name_emb, family_member_ids, all_persons_details):
    """
    Helper to compute features for one (input_person, candidate_family) pair.
    family_member_ids: list of PersonIDs in the candidate family.
    all_persons_details: dict with pre-loaded embeddings for all persons.
    """
    if not family_member_ids: return None

    member_face_embs = []
    member_name_embs = []

    for member_id in family_member_ids:
        member_details = all_persons_details.get(member_id)
        if member_details:
            if member_details['avg_face_emb'] is not None:
                member_face_embs.append(member_details['avg_face_emb'])
            if member_details['name_emb'] is not None:
                member_name_embs.append(member_details['name_emb'])

    if not member_face_embs:  # Need at least one family member with face embedding
        return None

    # Feature 1: Avg face embedding of the family
    avg_family_face_emb = np.mean(np.array(member_face_embs), axis=0)

    # Feature 2: Cosine similarity between input_face_emb and avg_family_face_emb
    sim_to_avg_family_face = calculate_cosine_similarity(input_face_emb, avg_family_face_emb)

    # Features 3,4,5: Min/Max/Avg cosine similarity (face) to family members
    face_sims_to_members = [calculate_cosine_similarity(input_face_emb, m_f_emb) for m_f_emb in member_face_embs]
    min_face_sim = np.min(face_sims_to_members) if face_sims_to_members else 0.0
    max_face_sim = np.max(face_sims_to_members) if face_sims_to_members else 0.0
    avg_face_sim = np.mean(face_sims_to_members) if face_sims_to_members else 0.0

    # Features for name similarity (if input_name_emb and member_name_embs exist)
    min_name_sim, max_name_sim, avg_name_sim = 0.0, 0.0, 0.0
    if input_name_emb is not None and member_name_embs:
        name_sims_to_members = [calculate_cosine_similarity(input_name_emb, m_n_emb) for m_n_emb in member_name_embs]
        min_name_sim = np.min(name_sims_to_members) if name_sims_to_members else 0.0
        max_name_sim = np.max(name_sims_to_members) if name_sims_to_members else 0.0
        avg_name_sim = np.mean(name_sims_to_members) if name_sims_to_members else 0.0

    # Feature: Number of members in the candidate family
    num_family_members = len(family_member_ids)

    # Concatenate all features
    # Order: sim_to_avg_family_face, min_face_sim, max_face_sim, avg_face_sim,
    #        min_name_sim, max_name_sim, avg_name_sim, num_family_members
    # Plus, diff between input_face_emb and avg_family_face_emb could be a feature too.
    # diff_face_emb = input_face_emb - avg_family_face_emb # vector feature
    # For simple classifiers, scalar features are easier.

    feature_vector = [
        sim_to_avg_family_face, min_face_sim, max_face_sim, avg_face_sim,
        min_name_sim, max_name_sim, avg_name_sim,
        float(num_family_members)
    ]
    return feature_vector


def train_model():
    X, y = generate_training_features_family()

    if X is None or y is None or len(X) == 0:
        logger.error("Feature generation failed or yielded no data. Classifier training aborted.")
        return

    logger.info(f"Generated {len(X)} samples for training. Feature vector length: {X.shape[1]}")
    logger.info(f"Class distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',  # Accuracy or F1 might also be good for binary
        random_seed=42,
        verbose=100
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

    logger.info("Model training complete.")
    model.save_model(config.FAMILY_CLASSIFIER_PATH)
    logger.info(f"Family membership classifier saved to {config.FAMILY_CLASSIFIER_PATH}")

    # Print feature importances
    try:
        feature_names = [
            "sim_to_avg_family_face", "min_face_sim_member", "max_face_sim_member", "avg_face_sim_member",
            "min_name_sim_member", "max_name_sim_member", "avg_name_sim_member",
            "num_family_members"
        ]
        importances = model.get_feature_importance()
        for name, imp in zip(feature_names, importances):
            logger.info(f"Feature: {name}, Importance: {imp:.4f}")
    except Exception as e:
        logger.warning(f"Could not get/print feature importances: {e}")


if __name__ == '__main__':
    # Ensure that average face embeddings pkl file is generated first for efficiency
    # To generate it (example, will be slow if run for the first time here):
    # if not os.path.exists(os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl")):
    #     logger.info("Pre-generating average face embeddings pkl file (can be slow)...")
    #     # The generate_training_features_family itself will create it if missing,
    #     # but it's better practice to have a dedicated script.
    #     # For this example, we let generate_training_features_family handle its creation.
    #     pass

    train_model()