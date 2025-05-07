# fiw_family_searcher/data_processing/database_setup.py
import sqlite3
import pandas as pd
import os
import faiss
import numpy as np
from tqdm import tqdm

from fiw_family_searcher import config
from fiw_family_searcher.utils.helpers import setup_logger, save_pickle
from fiw_family_searcher.data_processing import data_loader, embedder

logger = setup_logger(__name__)


def create_schema(conn):
    cursor = conn.cursor()

    # RIDs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS RIDs (
        RID INTEGER PRIMARY KEY,
        Label TEXT NOT NULL
    )""")

    # Families Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Families (
        FID TEXT PRIMARY KEY,
        FamilyRepName TEXT 
    )""")

    # Persons Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Persons (
        PersonID TEXT PRIMARY KEY, -- FID_MID
        FID TEXT NOT NULL,
        MID INTEGER NOT NULL,
        Name TEXT,
        Gender TEXT,
        NameEmbedding BLOB, -- Storing as BLOB, or can be kept only in FAISS
        FOREIGN KEY (FID) REFERENCES Families(FID)
    )""")

    # Photos Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Photos (
        PhotoID INTEGER PRIMARY KEY AUTOINCREMENT,
        PersonID TEXT NOT NULL,
        PhotoPath TEXT NOT NULL UNIQUE,
        -- FaceEmbedding BLOB, -- Not storing full embedding in SQL for performance with FAISS
        FOREIGN KEY (PersonID) REFERENCES Persons(PersonID)
    )""")

    # Relationships Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Relationships (
        RelID INTEGER PRIMARY KEY AUTOINCREMENT,
        Person1ID TEXT NOT NULL, -- Source Person
        Person2ID TEXT NOT NULL, -- Target Person
        RID INTEGER NOT NULL,    -- RelationshipTypeID
        FOREIGN KEY (Person1ID) REFERENCES Persons(PersonID),
        FOREIGN KEY (Person2ID) REFERENCES Persons(PersonID),
        FOREIGN KEY (RID) REFERENCES RIDs(RID)
    )""")

    conn.commit()
    logger.info("Database schema created/verified.")


def populate_rids_table(conn):
    df_rids = data_loader.load_rids()
    if not df_rids.empty:
        df_rids.to_sql("RIDs", conn, if_exists="replace", index=False)
        logger.info(f"Populated RIDs table with {len(df_rids)} entries.")


def populate_base_tables(conn):
    logger.info("Starting to populate base tables (Families, Persons, Photos, Relationships)...")

    # Load FIDs for Families table
    df_fids = data_loader.load_fids() # This will now have columns 'FID' and 'Label'
    if not df_fids.empty:
        df_fids_renamed = df_fids.rename(columns={"Label": "FamilyRepName"})
        df_fids_renamed.to_sql("Families", conn, if_exists="append", index=False)
        logger.info(f"Populated/Updated Families table with {len(df_fids_renamed)} entries.")
    else:
        logger.warning("FIW_FIDs.csv was empty or not loaded. Families table might be empty.")

    all_persons_data, all_relationships_data = data_loader.load_all_family_data()

    if not all_persons_data:
        logger.error("No person data loaded. Aborting further population.")
        return

    # Prepare data for Persons and Photos tables
    persons_for_sql = []
    photos_for_sql = []

    logger.info("Generating name embeddings for persons...")
    for person_data in tqdm(all_persons_data, desc="Processing persons for DB"):
        name_emb = embedder.get_name_embedding(person_data["Name"])
        persons_for_sql.append({
            "PersonID": person_data["PersonID"],
            "FID": person_data["FID"],
            "MID": person_data["MID"],
            "Name": person_data["Name"],
            "Gender": person_data["Gender"],
            "NameEmbedding": sqlite3.Binary(name_emb) if name_emb is not None else None
        })
        for photo_path in person_data["PhotoPaths"]:
            photos_for_sql.append({
                "PersonID": person_data["PersonID"],
                "PhotoPath": photo_path
            })

    df_persons = pd.DataFrame(persons_for_sql)
    df_photos = pd.DataFrame(photos_for_sql)

    if not df_persons.empty:
        df_persons.to_sql("Persons", conn, if_exists="append", index=False)
        logger.info(f"Populated/Updated Persons table with {len(df_persons)} entries.")

    if not df_photos.empty:
        # Handle potential unique constraint errors if run multiple times
        # A more robust way would be to insert one by one with try-except or INSERT OR IGNORE
        try:
            df_photos.to_sql("Photos", conn, if_exists="append", index=False)
            logger.info(f"Populated/Updated Photos table with {len(df_photos)} entries.")
        except sqlite3.IntegrityError as e:
            logger.warning(
                f"Integrity error while populating Photos (possibly due to rerunning): {e}. Some photo entries might already exist.")

    # Populate Relationships table
    if all_relationships_data:
        df_relationships = pd.DataFrame(all_relationships_data)
        df_relationships.to_sql("Relationships", conn, if_exists="append", index=False)
        logger.info(f"Populated/Updated Relationships table with {len(df_relationships)} entries.")

    conn.commit()
    logger.info("Base tables populated.")


def build_faiss_indexes():
    logger.info("Building FAISS indexes...")
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    # 1. Face Embeddings Index
    face_embeddings_list = []
    face_id_map_list = []  # Maps FAISS index to PersonID (can be multiple photos per person)

    # Get all photos and their PersonIDs
    cursor.execute("SELECT PhotoPath, PersonID FROM Photos")
    photo_records = cursor.fetchall()

    logger.info(f"Generating face embeddings for {len(photo_records)} photos...")
    for photo_path, person_id in tqdm(photo_records, desc="Face Embeddings"):
        abs_photo_path = photo_path  # Assuming paths stored are already absolute or resolvable
        # If paths are relative to PUBLIC_DIR or FIDS_FULL_DIR, prepend it.
        # Current data_loader stores absolute paths via glob.

        face_emb = embedder.get_face_embedding(abs_photo_path)
        if face_emb is not None:
            face_embeddings_list.append(face_emb)
            # Store PhotoID or PersonID? If PersonID, many FAISS entries map to one person.
            # For face search, we want to find similar *faces*, then map to person.
            # Let's map FAISS index to PersonID for simplicity.
            # Or better, PhotoID from DB, then get PersonID from PhotoID.
            # For now, PersonID directly for simpler mapping during search.
            face_id_map_list.append(person_id)

    if not face_embeddings_list:
        logger.warning("No face embeddings generated. FAISS face index will be empty.")
    else:
        face_embeddings_np = np.array(face_embeddings_list).astype('float32')
        if face_embeddings_np.shape[1] != config.FACE_EMBEDDING_DIM:
            logger.warning(
                f"Face embedding dimension mismatch! Expected {config.FACE_EMBEDDING_DIM}, Got {face_embeddings_np.shape[1]}. Check DeepFace model output.")
            # This could be an issue if config.FACE_EMBEDDING_DIM is not matching actual output
            # For robust solution, get dim from first embedding: actual_dim = face_embeddings_np.shape[1]

        actual_face_dim = face_embeddings_np.shape[1]
        face_index = faiss.IndexFlatL2(actual_face_dim)  # Using L2 distance
        # face_index = faiss.IndexFlatIP(actual_face_dim) # Or Inner Product (cosine similarity)
        face_index.add(face_embeddings_np)
        faiss.write_index(face_index, config.FACE_INDEX_PATH)
        save_pickle(face_id_map_list, config.FACE_ID_MAP_PATH)
        logger.info(
            f"FAISS face index built with {face_index.ntotal} embeddings and saved. Actual dim: {actual_face_dim}")

    # 2. Name Embeddings Index
    name_embeddings_list = []
    name_id_map_list = []  # Maps FAISS index to PersonID (one per person)

    cursor.execute("SELECT PersonID, NameEmbedding FROM Persons WHERE NameEmbedding IS NOT NULL")
    person_records_for_name = cursor.fetchall()

    logger.info(f"Loading name embeddings for {len(person_records_for_name)} persons...")
    for person_id, name_emb_blob in tqdm(person_records_for_name, desc="Name Embeddings"):
        name_emb = np.frombuffer(name_emb_blob, dtype=np.float32)
        name_embeddings_list.append(name_emb)
        name_id_map_list.append(person_id)

    if not name_embeddings_list:
        logger.warning("No name embeddings found. FAISS name index will be empty.")
    else:
        name_embeddings_np = np.array(name_embeddings_list).astype('float32')
        if name_embeddings_np.shape[1] != config.NAME_EMBEDDING_DIM:
            logger.warning(
                f"Name embedding dimension mismatch! Expected {config.NAME_EMBEDDING_DIM}, Got {name_embeddings_np.shape[1]}.")

        actual_name_dim = name_embeddings_np.shape[1]
        name_index = faiss.IndexFlatL2(actual_name_dim)  # L2 distance
        # name_index = faiss.IndexFlatIP(actual_name_dim) # Or Inner Product
        name_index.add(name_embeddings_np)
        faiss.write_index(name_index, config.NAME_INDEX_PATH)
        save_pickle(name_id_map_list, config.NAME_ID_MAP_PATH)
        logger.info(
            f"FAISS name index built with {name_index.ntotal} embeddings and saved. Actual dim: {actual_name_dim}")

    conn.close()


def setup_database_and_indexes(force_rebuild=False):
    if os.path.exists(config.DB_PATH) and \
            os.path.exists(config.FACE_INDEX_PATH) and \
            os.path.exists(config.NAME_INDEX_PATH) and \
            not force_rebuild:
        logger.info("Database and FAISS indexes already exist. Skipping setup.")
        logger.info(f"DB: {config.DB_PATH}")
        logger.info(f"Face Index: {config.FACE_INDEX_PATH} ({os.path.getsize(config.FACE_INDEX_PATH)} bytes)")
        logger.info(f"Name Index: {config.NAME_INDEX_PATH} ({os.path.getsize(config.NAME_INDEX_PATH)} bytes)")
        return

    logger.info("Starting database and FAISS index setup...")

    # Ensure embedding/model directories exist
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Delete existing files if force_rebuild
    if force_rebuild:
        for path in [config.DB_PATH, config.FACE_INDEX_PATH, config.NAME_INDEX_PATH,
                     config.FACE_ID_MAP_PATH, config.NAME_ID_MAP_PATH]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed existing file: {path}")

    conn = sqlite3.connect(config.DB_PATH)
    create_schema(conn)
    populate_rids_table(conn)  # Should be idempotent if using "replace"
    populate_base_tables(conn)  # Uses "append", so safe to rerun but might create duplicates if PK not well handled.
    # PersonID is PK, so it should be fine. PhotoPath is UNIQUE.
    conn.close()  # Close connection before FAISS build which opens its own

    build_faiss_indexes()
    logger.info("Database and FAISS index setup complete.")


if __name__ == '__main__':
    # This will take a very long time for the full dataset
    # Set force_rebuild=True to re-generate everything
    # It's advisable to run this once and then comment out or use force_rebuild=False
    logger.info("Running database setup script...")
    setup_database_and_indexes(force_rebuild=False)
    # Example: setup_database_and_indexes(force_rebuild=True) # To force a full rebuild
    logger.info("Database setup script finished.")