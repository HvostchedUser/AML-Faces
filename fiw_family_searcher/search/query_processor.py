# fiw_family_searcher/search/query_processor.py
import numpy as np
import sqlite3
from catboost import CatBoostClassifier
import os

from fiw_family_searcher import config
from fiw_family_searcher.utils.helpers import setup_logger, load_pickle, calculate_cosine_similarity
from fiw_family_searcher.data_processing import embedder
from fiw_family_searcher.search.faiss_search import FaissSearcher
from fiw_family_searcher.search.fusion import reciprocal_rank_fusion
# For training family classifier, _extract_family_features was used. Need similar for inference.
from fiw_family_searcher.training.train_family_classifier import _extract_family_features as extract_family_clf_features

# We need access to all_persons_details for _extract_family_features
# This indicates _extract_family_features might need to be more general or refactored.

logger = setup_logger(__name__)


class QueryProcessor:
    def __init__(self):
        self.faiss_searcher = FaissSearcher()
        self.family_classifier = None
        self.relationship_classifier = None
        self.rid_labels = {}  # {RID: Label}
        self._load_models()
        self._load_rid_labels()

        # For _extract_family_features, we need all_persons_details. Load it once.
        self._all_persons_details_for_clf = self._load_all_persons_details_for_clf()
        # For relationship classifier, we need person_avg_face_embeddings
        self._person_avg_face_embeddings_for_rel_clf = self._load_person_avg_face_embeddings()

    def _load_models(self):
        if os.path.exists(config.FAMILY_CLASSIFIER_PATH):
            self.family_classifier = CatBoostClassifier()
            self.family_classifier.load_model(config.FAMILY_CLASSIFIER_PATH)
            logger.info("Family membership classifier loaded.")
        else:
            logger.warning("Family membership classifier model not found. Predictions will be limited.")

        if os.path.exists(config.RELATIONSHIP_CLASSIFIER_PATH):
            self.relationship_classifier = CatBoostClassifier()
            self.relationship_classifier.load_model(config.RELATIONSHIP_CLASSIFIER_PATH)
            logger.info("Relationship classifier loaded.")
        else:
            logger.warning("Relationship classifier model not found. Relationship predictions will be limited.")

    def _load_rid_labels(self):
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT RID, Label FROM RIDs")
            for rid, label in cursor.fetchall():
                self.rid_labels[rid] = label
            conn.close()
            logger.info(f"Loaded {len(self.rid_labels)} RID labels.")
        except Exception as e:
            logger.error(f"Error loading RID labels: {e}")

    def _load_all_persons_details_for_clf(self):
        """Loads necessary person details (embeddings, FID) for family classifier feature extraction."""
        # This is similar to the logic in train_family_classifier.py
        # It's needed by _extract_family_features
        details = {}
        avg_face_embs_path = os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl")
        person_avg_face_embeddings = {}
        if os.path.exists(avg_face_embs_path):
            person_avg_face_embeddings = load_pickle(avg_face_embs_path)
        else:
            logger.warning(f"File not found: {avg_face_embs_path}. Family classification features might be impaired.")

        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT PersonID, NameEmbedding, FID FROM Persons")
            for pid, name_emb_blob, fid_val in cursor.fetchall():
                name_e = np.frombuffer(name_emb_blob, dtype=np.float32) if name_emb_blob else None
                avg_face_e = person_avg_face_embeddings.get(pid)
                details[pid] = {'name_emb': name_e, 'avg_face_emb': avg_face_e, 'fid': fid_val}
            conn.close()
            logger.info(f"Loaded person details for {len(details)} persons for family classifier.")
        except Exception as e:
            logger.error(f"Error loading person details for classifier: {e}")
        return details

    def _load_person_avg_face_embeddings(self):
        """Loads average face embeddings needed for relationship classifier."""
        avg_face_embs_path = os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl")
        if os.path.exists(avg_face_embs_path):
            logger.info(f"Loading average face embeddings from {avg_face_embs_path} for rel classifier.")
            return load_pickle(avg_face_embs_path)
        else:
            logger.warning(f"{avg_face_embs_path} not found. Relationship predictions might be impaired.")
            return {}

    def _get_family_members_from_db(self, fid):
        """ Fetches PersonIDs of members for a given FID """
        members = []
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT PersonID FROM Persons WHERE FID = ?", (fid,))
            members = [row[0] for row in cursor.fetchall()]
            conn.close()
        except Exception as e:
            logger.error(f"Error fetching members for FID {fid}: {e}")
        return members

    def _get_person_details_from_db(self, person_ids):
        """Fetches name and FID for a list of PersonIDs."""
        details = {}  # {PersonID: {'Name': name, 'FID': fid}}
        if not person_ids: return details
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            # Create a placeholder string for IN clause
            placeholders = ','.join('?' for _ in person_ids)
            query = f"SELECT PersonID, Name, FID FROM Persons WHERE PersonID IN ({placeholders})"
            cursor.execute(query, person_ids)
            for pid, name, fid_val in cursor.fetchall():
                details[pid] = {'Name': name, 'FID': fid_val}
            conn.close()
        except Exception as e:
            logger.error(f"Error fetching person details: {e}")
        return details

    def find_family_and_relations(self, input_photo_path, input_name_str=None, top_n_families=3):
        """
        Main query processing pipeline.
        """
        results = {
            "input_photo": input_photo_path,
            "input_name": input_name_str,
            "identified_person": None,  # If input matches an existing person strongly
            "candidate_families": [],  # List of {family_id, probability, members_relations}
        }

        # 1. Get Embeddings for input
        input_face_emb = embedder.get_face_embedding(input_photo_path)
        if input_face_emb is None:
            logger.error("Could not generate face embedding for input photo. Aborting.")
            results["error"] = "Failed to process input photo."
            return results

        input_name_emb = None
        if input_name_str:
            input_name_emb = embedder.get_name_embedding(input_name_str)

        # 2. Initial FAISS Search
        face_search_person_ids, face_search_distances = self.faiss_searcher.search_face(input_face_emb)

        # Convert L2 distances to similarity scores (0 to 1, higher is better)
        # Simple conversion: score = 1 / (1 + distance). Max distance can be large.
        # Normalized score: score = exp(-distance / sigma), sigma is a scaling factor (e.g. mean distance)
        # Or, for ArcFace, cosine distance is 1-cos_sim. If FAISS stores L2 on normalized embs, L2^2 = 2*(1-cos_sim)
        # So, cos_sim = 1 - (L2_dist^2 / 2). Ensure embeddings are normalized for this.
        # DeepFace ArcFace embeddings are typically normalized.
        face_search_scores = [1 - (d ** 2 / 2) if d is not None else 0 for d in
                              face_search_distances]  # Cosine sim from L2
        face_results_scored = list(zip(face_search_person_ids, face_search_scores))

        ranked_lists_for_fusion = [face_results_scored]

        if input_name_emb is not None:
            name_search_person_ids, name_search_distances = self.faiss_searcher.search_name(input_name_emb)
            name_search_scores = [1 - (d ** 2 / 2) if d is not None else 0 for d in
                                  name_search_distances]  # Cosine sim from L2
            name_results_scored = list(zip(name_search_person_ids, name_search_scores))
            ranked_lists_for_fusion.append(name_results_scored)

        # 3. Fusion
        fused_results = reciprocal_rank_fusion(ranked_lists_for_fusion, k=config.RRF_K)
        # fused_results is [(PersonID, rrf_score), ...]

        if not fused_results:
            logger.warning("No results after FAISS search and fusion.")
            results["message"] = "No potential matches found in the database."
            return results

        # Check for strong direct match
        # Strongest match from face search
        if face_results_scored and face_results_scored[0][1] > config.STRONG_DIRECT_MATCH_SIMILARITY_THRESHOLD:
            strongest_match_pid = face_results_scored[0][0]
            strongest_match_details = self._get_person_details_from_db([strongest_match_pid]).get(strongest_match_pid)
            if strongest_match_details:
                results["identified_person"] = {
                    "PersonID": strongest_match_pid,
                    "Name": strongest_match_details["Name"],
                    "FID": strongest_match_details["FID"],
                    "face_similarity": face_results_scored[0][1]
                }
                logger.info(
                    f"Potential strong direct match: {strongest_match_pid} with face sim {face_results_scored[0][1]:.4f}")

        # 4. Candidate Family Identification
        candidate_fids = set()
        # Take top N distinct persons from fused results to get candidate families
        distinct_persons_for_families = []
        seen_pids_for_families = set()
        for pid, _ in fused_results:
            if pid not in seen_pids_for_families:
                distinct_persons_for_families.append(pid)
                seen_pids_for_families.add(pid)
            if len(distinct_persons_for_families) >= config.FAISS_SEARCH_TOP_K:  # e.g. top 20 distinct persons
                break

        person_details_map = self._get_person_details_from_db(distinct_persons_for_families)
        for pid in distinct_persons_for_families:
            detail = person_details_map.get(pid)
            if detail:
                candidate_fids.add(detail['FID'])

        logger.info(f"Candidate FIDs from initial search: {candidate_fids}")

        # 5. Family Membership Classification
        family_membership_scores = []  # [(FID, probability_belongs)]
        if self.family_classifier and self._all_persons_details_for_clf:
            for fid in candidate_fids:
                family_member_ids = self._get_family_members_from_db(fid)
                if not family_member_ids: continue

                # _extract_family_features expects all_persons_details, which we loaded
                # Need to ensure input_face_emb and input_name_emb are correctly passed
                features = extract_family_clf_features(
                    input_face_emb, input_name_emb,
                    family_member_ids, self._all_persons_details_for_clf
                )
                if features:
                    prob_belongs = self.family_classifier.predict_proba([features])[0][1]  # Prob of class 1
                    family_membership_scores.append((fid, prob_belongs))

            # Sort families by probability of belonging
            family_membership_scores.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Family membership probabilities: {family_membership_scores[:top_n_families]}")
        else:
            logger.warning(
                "Family classifier not available or details missing. Using FIDs from top fused results directly.")
            # Fallback: use FIDs from top fused results directly, no probabilities
            for fid in list(candidate_fids)[:top_n_families]:  # Take first few unique FIDs
                family_membership_scores.append((fid, -1.0))  # -1 indicates no probability

        # 6. Relationship Prediction for Top N Families
        for i, (fid, prob_belongs) in enumerate(family_membership_scores):
            if i >= top_n_families: break  # Process only top_n_families

            if prob_belongs < config.FAMILY_MEMBERSHIP_THRESHOLD and prob_belongs != -1.0:  # -1 means fallback
                logger.info(
                    f"Skipping family {fid}, prob {prob_belongs:.4f} below threshold {config.FAMILY_MEMBERSHIP_THRESHOLD}")
                continue

            family_info = {"family_id": fid,
                           "probability_belongs": f"{prob_belongs:.4f}" if prob_belongs != -1.0 else "N/A",
                           "members": []}
            current_family_members = self._get_family_members_from_db(fid)
            member_details = self._get_person_details_from_db(current_family_members)

            if self.relationship_classifier and self._person_avg_face_embeddings_for_rel_clf:
                for member_pid in current_family_members:
                    member_data = {"person_id": member_pid,
                                   "name": member_details.get(member_pid, {}).get('Name', 'Unknown')}

                    member_face_emb = self._person_avg_face_embeddings_for_rel_clf.get(member_pid)
                    if member_face_emb is None:
                        member_data["relationship_to_input"] = "Error: Member face embedding missing"
                        family_info["members"].append(member_data)
                        continue

                    # Prepare features for relationship classifier: [input_face, member_face, diff, prod]
                    diff_emb = np.abs(input_face_emb - member_face_emb)
                    prod_emb = input_face_emb * member_face_emb
                    rel_features = np.concatenate((input_face_emb, member_face_emb, diff_emb, prod_emb))

                    pred_probs = self.relationship_classifier.predict_proba([rel_features])[0]
                    predicted_rid_idx = np.argmax(pred_probs)
                    predicted_rid = self.relationship_classifier.classes_[predicted_rid_idx]  # Get actual RID value

                    relationship_label = self.rid_labels.get(predicted_rid, f"Unknown RID {predicted_rid}")
                    confidence = pred_probs[predicted_rid_idx]

                    member_data["relationship_to_input"] = f"{relationship_label} (Confidence: {confidence:.2f})"
                    family_info["members"].append(member_data)
            else:
                family_info["message"] = "Relationship classifier not available or embeddings missing."
                for member_pid in current_family_members:
                    member_data = {"person_id": member_pid,
                                   "name": member_details.get(member_pid, {}).get('Name', 'Unknown'),
                                   "relationship_to_input": "N/A"}
                    family_info["members"].append(member_data)

            results["candidate_families"].append(family_info)

        return results