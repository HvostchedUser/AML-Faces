# fiw_family_searcher/search/query_processor.py
import numpy as np
import sqlite3
from catboost import CatBoostClassifier
import os
from deepface import DeepFace  # Added DeepFace import

from fiw_family_searcher import config
from fiw_family_searcher.utils.helpers import setup_logger, \
    load_pickle  # Removed calculate_cosine_similarity as it's not directly used here
from fiw_family_searcher.data_processing import embedder  # embedder.get_name_embedding is used
from fiw_family_searcher.search.faiss_search import FaissSearcher
from fiw_family_searcher.search.fusion import reciprocal_rank_fusion
from fiw_family_searcher.training.train_family_classifier import _extract_family_features as extract_family_clf_features
from fiw_family_searcher.explainability.similarity_explainer import SimilarityExplainer

logger = setup_logger(__name__)


class QueryProcessor:
    def __init__(self):
        self.faiss_searcher = FaissSearcher()
        self.family_classifier = None
        self.relationship_classifier = None
        self.rid_labels = {}
        self._load_models()
        self._load_rid_labels()

        self._all_persons_details_for_clf = self._load_all_persons_details_for_clf()
        self._person_avg_face_embeddings_for_rel_clf = self._load_person_avg_face_embeddings()

        self.similarity_explainer = None
        try:
            self.similarity_explainer = SimilarityExplainer()
            if self.similarity_explainer.keras_model is None:
                logger.warning(
                    "SimilarityExplainer initialized, but its core Keras model failed to load. Explanations will not be available.")
                self.similarity_explainer = None
            else:
                logger.info("SimilarityExplainer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SimilarityExplainer: {e}", exc_info=True)
            self.similarity_explainer = None

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
        avg_face_embs_path = os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl")
        if os.path.exists(avg_face_embs_path):
            logger.info(f"Loading average face embeddings from {avg_face_embs_path} for rel classifier.")
            return load_pickle(avg_face_embs_path)
        else:
            logger.warning(f"{avg_face_embs_path} not found. Relationship predictions might be impaired.")
            return {}

    def _get_family_members_from_db(self, fid):
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
        details = {}
        if not person_ids: return details
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            placeholders = ','.join('?' for _ in person_ids)
            query = f"""
                SELECT p.PersonID, p.Name, p.FID, ph.PhotoPath
                FROM Persons p
                LEFT JOIN (
                    SELECT PersonID, MIN(PhotoPath) as PhotoPath 
                    FROM Photos 
                    WHERE PersonID IN ({placeholders})
                    GROUP BY PersonID
                ) ph ON p.PersonID = ph.PersonID
                WHERE p.PersonID IN ({placeholders})
            """
            # The parameters list needs to be duplicated because placeholders appear twice
            cursor.execute(query, person_ids + person_ids)
            for pid, name, fid_val, photo_path in cursor.fetchall():
                details[pid] = {'Name': name, 'FID': fid_val, 'PhotoPath': photo_path}
            conn.close()
        except Exception as e:
            logger.error(f"Error fetching person details with photo path: {e}")
        return details

    def _clean_family_rep_name(self, rep_name_str):
        if not rep_name_str:
            return "Unknown Family"
        if '.' in rep_name_str:
            parts = rep_name_str.split('.')
            if len(parts) > 2 and "family" in parts:  # e.g. "obama.michelle.family"
                # Capitalize all parts except 'family' if it's just a descriptor
                name_parts = [word.capitalize() for word in parts if word.lower() != "family"]
                return ' '.join(name_parts) + " Family" if "family" in parts else ' '.join(name_parts)

            surname = parts[0]
            return surname.capitalize()
        return rep_name_str.replace('_', ' ').title()

    def _get_family_details_from_db(self, fid):
        family_details = {}
        try:
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT FID, FamilyRepName FROM Families WHERE FID = ?", (fid,))
            row = cursor.fetchone()
            if row:
                original_rep_name = row[1]
                cleaned_name = self._clean_family_rep_name(original_rep_name)
                family_details = {'FID': row[0], 'FamilyRepName': original_rep_name, 'CleanedName': cleaned_name}
            else:  # Fallback if FID not in Families table (should not happen with good DB setup)
                family_details = {'FID': fid, 'FamilyRepName': fid, 'CleanedName': fid}  # Use FID as name
            conn.close()
        except Exception as e:
            logger.error(f"Error fetching details for FID {fid}: {e}")
            family_details = {'FID': fid, 'FamilyRepName': fid, 'CleanedName': fid}
        return family_details

    def find_family_and_relations(self, input_photo_path, input_name_str=None, top_n_families=3):
        results = {
            "input_photo": input_photo_path,  # Original path for reference/explainer
            "input_name": input_name_str,
            "identified_person": None,
            "candidate_families": [],
        }

        input_face_emb = None
        try:
            logger.info(f"Extracting face from user uploaded photo: {input_photo_path}")
            # Use a faster detector like 'ssd' or 'retinaface' for user uploads
            extracted_faces = DeepFace.extract_faces(
                img_path=input_photo_path,
                detector_backend='ssd',  # Faster detector
                enforce_detection=True,  # Must find a face
                align=True  # Align the face
            )
            if not extracted_faces or 'face' not in extracted_faces[0]:
                logger.error("No face detected in the uploaded image.")
                results["error"] = "No face detected in the uploaded image. Please use a clear photo of a face."
                return results

            # Use the first detected face (NumPy array, RGB, [0,1] range)
            cropped_face_array = extracted_faces[0]['face']
            logger.info("Face extracted successfully. Generating embedding from cropped face.")

            # Generate embedding from the cropped face array
            # detector_backend='skip' is crucial when img_path is a numpy array
            embedding_objs = DeepFace.represent(
                img_path=cropped_face_array,
                model_name=config.FACE_EMBEDDING_MODEL_DEEPFACE,
                enforce_detection=False,  # Already detected
                detector_backend='skip'
            )
            if embedding_objs and len(embedding_objs) > 0:
                input_face_emb = np.array(embedding_objs[0]['embedding'], dtype=np.float32)
            else:
                logger.error("Could not generate face embedding from extracted face. Aborting.")
                results["error"] = "Failed to process the extracted face from input photo."
                return results

        except Exception as e:
            # Catch specific DeepFace no face detected error
            if "Face could not be detected" in str(e) or "No face detected" in str(e):
                logger.error(f"No face detected in {input_photo_path} by DeepFace.extract_faces. Error: {e}")
                results["error"] = "No face detected in the uploaded image. Please use a clear photo of a face."
            else:
                logger.error(f"Error during face extraction or embedding for input photo: {e}", exc_info=True)
                results["error"] = "An error occurred while processing your photo."
            return results

        input_name_emb = None
        if input_name_str:
            input_name_emb = embedder.get_name_embedding(input_name_str)  # From original embedder

        face_search_person_ids, face_search_distances = self.faiss_searcher.search_face(input_face_emb)
        face_search_scores = [1 - (d ** 2 / 2) if d is not None else 0 for d in face_search_distances]
        face_results_scored = list(zip(face_search_person_ids, face_search_scores))

        ranked_lists_for_fusion = [face_results_scored]

        if input_name_emb is not None:
            name_search_person_ids, name_search_distances = self.faiss_searcher.search_name(input_name_emb)
            name_search_scores = [1 - (d ** 2 / 2) if d is not None else 0 for d in name_search_distances]
            name_results_scored = list(zip(name_search_person_ids, name_search_scores))
            ranked_lists_for_fusion.append(name_results_scored)

        fused_results = reciprocal_rank_fusion(ranked_lists_for_fusion, k=config.RRF_K)

        if not fused_results:
            logger.warning("No results after FAISS search and fusion.")
            results["message"] = "No potential matches found in the database."
            return results

        if face_results_scored and face_results_scored[0][1] > config.STRONG_DIRECT_MATCH_SIMILARITY_THRESHOLD:
            strongest_match_pid = face_results_scored[0][0]
            strongest_match_details = self._get_person_details_from_db([strongest_match_pid]).get(strongest_match_pid)
            if strongest_match_details:
                results["identified_person"] = {
                    "PersonID": strongest_match_pid,
                    "Name": strongest_match_details["Name"],
                    "FID": strongest_match_details["FID"],
                    "face_similarity": float(f"{face_results_scored[0][1]:.4f}"),  # Format as float string
                    "photo_path": strongest_match_details.get("PhotoPath")
                }
                logger.info(
                    f"Potential strong direct match: {strongest_match_pid} with face sim {face_results_scored[0][1]:.4f}")

        candidate_fids = set()
        distinct_persons_for_families = []
        seen_pids_for_families = set()
        for pid, _ in fused_results:
            if pid not in seen_pids_for_families:
                distinct_persons_for_families.append(pid)
                seen_pids_for_families.add(pid)
            if len(distinct_persons_for_families) >= config.FAISS_SEARCH_TOP_K:
                break

        person_details_map = self._get_person_details_from_db(distinct_persons_for_families)
        for pid in distinct_persons_for_families:
            detail = person_details_map.get(pid)
            if detail and 'FID' in detail:  # Ensure FID exists
                candidate_fids.add(detail['FID'])

        logger.info(f"Candidate FIDs from initial search: {candidate_fids}")

        family_membership_scores = []
        if self.family_classifier and self._all_persons_details_for_clf:
            for fid_val in candidate_fids:  # Iterate over fid_val from the set
                family_member_ids = self._get_family_members_from_db(fid_val)
                if not family_member_ids: continue

                features = extract_family_clf_features(
                    input_face_emb, input_name_emb,
                    family_member_ids, self._all_persons_details_for_clf
                )
                if features is not None and len(features) > 0:
                    prob_belongs = self.family_classifier.predict_proba([features])[0][1]
                    family_membership_scores.append((fid_val, prob_belongs))
                else:
                    logger.debug(f"Could not extract features for family {fid_val}, skipping membership score.")

            family_membership_scores.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Family membership probabilities: {family_membership_scores[:top_n_families]}")
        else:
            logger.warning(
                "Family classifier not available or details missing. Using FIDs from top fused results directly.")
            for fid_val in list(candidate_fids)[:top_n_families]:
                family_membership_scores.append((fid_val, -1.0))

        for i, (fid, prob_belongs) in enumerate(family_membership_scores):
            if i >= top_n_families: break
            if prob_belongs < config.FAMILY_MEMBERSHIP_THRESHOLD and prob_belongs != -1.0:
                logger.info(
                    f"Skipping family {fid}, prob {prob_belongs:.4f} below threshold {config.FAMILY_MEMBERSHIP_THRESHOLD}")
                continue

            family_db_details = self._get_family_details_from_db(fid)
            family_display_name = family_db_details.get('CleanedName', fid)

            family_info = {
                "family_id": fid,
                "family_display_name": family_display_name,
                "probability_belongs": f"{prob_belongs:.4f}" if prob_belongs != -1.0 else "N/A",
                "members": []
            }

            current_family_members_pids = self._get_family_members_from_db(fid)
            current_family_member_details = self._get_person_details_from_db(current_family_members_pids)

            if self.relationship_classifier and self._person_avg_face_embeddings_for_rel_clf:
                for member_pid in current_family_members_pids:
                    member_data = {"person_id": member_pid}
                    member_db_details = current_family_member_details.get(member_pid, {})
                    member_data["name"] = member_db_details.get('Name', 'Unknown')
                    member_photo_path_abs = member_db_details.get('PhotoPath')
                    member_data["photo_path_abs"] = member_photo_path_abs

                    if member_photo_path_abs:
                        try:
                            abs_fids_full_dir = os.path.abspath(config.FIDS_FULL_DIR)
                            if os.path.commonpath([member_photo_path_abs, abs_fids_full_dir]) == abs_fids_full_dir:
                                relative_photo_path = os.path.relpath(member_photo_path_abs, abs_fids_full_dir)
                                member_data["photo_url"] = f"/member_images/{relative_photo_path.replace(os.sep, '/')}"
                            else:
                                member_data["photo_url"] = None
                                logger.warning(
                                    f"Photo path {member_photo_path_abs} for {member_pid} is not relative to {abs_fids_full_dir}")
                        except ValueError:
                            member_data["photo_url"] = None
                            logger.warning(
                                f"Could not determine relative path for {member_photo_path_abs} against {abs_fids_full_dir}")
                    else:
                        member_data["photo_url"] = None

                    member_face_emb = self._person_avg_face_embeddings_for_rel_clf.get(member_pid)
                    if member_face_emb is None or input_face_emb is None:
                        member_data["relationship_to_input"] = "Error: Face embedding missing for query or member."
                        family_info["members"].append(member_data)
                        continue

                    diff_emb = np.abs(input_face_emb - member_face_emb)
                    prod_emb = input_face_emb * member_face_emb
                    rel_features = np.concatenate((input_face_emb, member_face_emb, diff_emb, prod_emb))

                    pred_probs = self.relationship_classifier.predict_proba([rel_features])[0]
                    predicted_rid_idx = np.argmax(pred_probs)
                    if hasattr(self.relationship_classifier, 'classes_') and predicted_rid_idx < len(
                            self.relationship_classifier.classes_):
                        predicted_rid = self.relationship_classifier.classes_[predicted_rid_idx]
                        relationship_label = self.rid_labels.get(predicted_rid, f"Unknown RID {predicted_rid}")
                        confidence = pred_probs[predicted_rid_idx]
                        member_data["relationship_to_input"] = f"{relationship_label} (Confidence: {confidence:.2f})"
                    else:
                        member_data["relationship_to_input"] = "Error: Relationship prediction failed."
                        logger.error(f"Rel classifier classes_ missing or index out of bounds for member {member_pid}")
                    family_info["members"].append(member_data)
            else:  # Fallback if relationship classifier is not available
                family_info["message"] = "Relationship classifier not available or embeddings missing."
                for member_pid in current_family_members_pids:
                    member_db_details = current_family_member_details.get(member_pid, {})
                    member_photo_path_abs = member_db_details.get('PhotoPath')
                    photo_url = None
                    if member_photo_path_abs:
                        try:
                            abs_fids_full_dir = os.path.abspath(config.FIDS_FULL_DIR)
                            if os.path.commonpath([member_photo_path_abs, abs_fids_full_dir]) == abs_fids_full_dir:
                                relative_photo_path = os.path.relpath(member_photo_path_abs, abs_fids_full_dir)
                                photo_url = f"/member_images/{relative_photo_path.replace(os.sep, '/')}"
                        except ValueError:
                            pass
                    family_info["members"].append({
                        "person_id": member_pid,
                        "name": member_db_details.get('Name', 'Unknown'),
                        "photo_url": photo_url,
                        "photo_path_abs": member_photo_path_abs,
                        "relationship_to_input": "N/A"
                    })
            results["candidate_families"].append(family_info)
        return results

    def get_similarity_explanation(self, query_photo_path, member_photo_path):
        if self.similarity_explainer is None:
            logger.warning("SimilarityExplainer not available. Cannot generate explanations.")
            return None, None

        if not query_photo_path or not os.path.exists(query_photo_path):
            logger.error(f"Query photo path invalid for explanation: {query_photo_path}")
            return None, None
        if not member_photo_path or not os.path.exists(member_photo_path):
            logger.error(f"Member photo path invalid for explanation: {member_photo_path}")
            return None, None

        logger.info(
            f"Generating SHAP explanation for query: {os.path.basename(query_photo_path)} and member: {os.path.basename(member_photo_path)}")

        try:
            heatmap_query_url, heatmap_member_url = self.similarity_explainer.explain_similarity(
                query_photo_path,
                member_photo_path
            )
            return heatmap_query_url, heatmap_member_url
        except Exception as e:
            logger.error(f"Exception during similarity explanation: {e}", exc_info=True)
            return None, None