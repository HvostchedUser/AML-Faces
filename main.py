import argparse
import os
import json

from fiw_family_searcher.utils.helpers import setup_logger
from fiw_family_searcher.data_processing.database_setup import setup_database_and_indexes
from fiw_family_searcher.training.train_family_classifier import train_model as train_family_clf
from fiw_family_searcher.training.train_relationship_classifier import train_model as train_rel_clf
from fiw_family_searcher.search.query_processor import QueryProcessor
from fiw_family_searcher import config

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="FIW Family Searcher System")
    parser.add_argument(
        "action",
        choices=["setup", "train_family_clf", "train_rel_clf", "query", "full_setup_train"],
        help="Action to perform: "
             "'setup' (builds DB and FAISS indexes), "
             "'train_family_clf' (trains family membership classifier), "
             "'train_rel_clf' (trains relationship classifier), "
             "'query' (runs a search query), "
             "'full_setup_train' (runs setup, then both trainings)"
    )
    parser.add_argument("--photo", type=str, help="Path to the input photo for query action.")
    parser.add_argument("--name", type=str, help="Optional name string for query action (e.g., 'John Doe').")
    parser.add_argument("--force_rebuild_db", action="store_true",
                        help="Force rebuild of DB and FAISS indexes during setup.")

    args = parser.parse_args()

    if args.action == "setup":
        logger.info("Starting database and FAISS index setup...")
        setup_database_and_indexes(force_rebuild=args.force_rebuild_db)
        logger.info("Setup complete.")
        # Crucial: After setup, generate the person_avg_face_embeddings.pkl file
        # This can be done by slightly modifying train_family_classifier.py or adding a new script
        # For now, let's indicate it's a manual next step or done by train_family_clf
        logger.info("IMPORTANT: Ensure 'person_avg_face_embeddings.pkl' is generated. "
                    "Running 'train_family_clf' action will attempt to create it if missing.")

    elif args.action == "train_family_clf":
        logger.info("Starting family membership classifier training...")
        # This script will attempt to create person_avg_face_embeddings.pkl if missing
        train_family_clf()
        logger.info("Family membership classifier training complete.")

    elif args.action == "train_rel_clf":
        # This script depends on person_avg_face_embeddings.pkl
        avg_face_embs_path = os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl")
        if not os.path.exists(avg_face_embs_path):
            logger.error(f"{avg_face_embs_path} not found. Please run 'train_family_clf' first to generate it.")
            return
        logger.info("Starting relationship classifier training...")
        train_rel_clf()
        logger.info("Relationship classifier training complete.")

    elif args.action == "full_setup_train":
        logger.info("--- Starting Full Setup and Training ---")
        logger.info("Step 1: Database and FAISS index setup...")
        setup_database_and_indexes(force_rebuild=args.force_rebuild_db)
        logger.info("Setup complete.")

        logger.info("Step 2: Training family membership classifier (will also generate avg face embeddings pkl)...")
        train_family_clf()
        logger.info("Family membership classifier training complete.")

        avg_face_embs_path = os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl")
        if not os.path.exists(avg_face_embs_path):
            logger.error(
                f"{avg_face_embs_path} not found after family_clf training. Cannot proceed to rel_clf training.")
            return

        logger.info("Step 3: Training relationship classifier...")
        train_rel_clf()
        logger.info("Relationship classifier training complete.")
        logger.info("--- Full Setup and Training Finished ---")


    elif args.action == "query":
        if not args.photo:
            logger.error("Photo path (--photo) is required for query action.")
            return
        if not os.path.exists(args.photo):
            logger.error(f"Photo path does not exist: {args.photo}")
            return

        # Ensure models and helper files (like pkl embeddings) exist
        required_files_for_query = [
            config.FAMILY_CLASSIFIER_PATH,
            config.RELATIONSHIP_CLASSIFIER_PATH,
            os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl"),
            config.FACE_INDEX_PATH, config.NAME_INDEX_PATH,  # FAISS indexes
            config.FACE_ID_MAP_PATH, config.NAME_ID_MAP_PATH
        ]
        missing_files = [f for f in required_files_for_query if not os.path.exists(f)]
        if missing_files:
            logger.error(f"Query cannot run. Missing required files: {missing_files}")
            logger.error(
                "Please run 'setup' and then 'train_family_clf' and 'train_rel_clf' actions first, or 'full_setup_train'.")
            return

        logger.info(f"Performing query for photo: {args.photo}" + (f" and name: {args.name}" if args.name else ""))

        query_processor = QueryProcessor()  # Loads models and indexes
        results = query_processor.find_family_and_relations(args.photo, args.name)

        logger.info("\n--- Query Results ---")
        print(json.dumps(results, indent=2))
        logger.info("--- End of Query Results ---")

    else:
        logger.error(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main()