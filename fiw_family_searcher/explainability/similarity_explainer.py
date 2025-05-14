# fiw_family_searcher/explainability/similarity_explainer.py
import shap
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from PIL import Image
import os
import uuid
import matplotlib

matplotlib.use('Agg')  # Explicitly set backend for Matplotlib before importing pyplot
import matplotlib.pyplot as plt
import logging
import time  # For timing sections

from fiw_family_searcher import config
from fiw_family_searcher.utils.helpers import setup_logger

logger = setup_logger(__name__)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


class SimilarityExplainer:
    def __init__(self):
        self.model_name = config.FACE_EMBEDDING_MODEL_DEEPFACE
        self.keras_model = None
        self.target_size = None
        try:
            logger.info("Initializing SimilarityExplainer...")
            init_start_time = time.time()

            built_object = DeepFace.build_model(self.model_name)

            if hasattr(built_object, 'predict') and callable(getattr(built_object, 'predict')):
                self.keras_model = built_object
            elif hasattr(built_object, 'model') and hasattr(built_object.model, 'predict') and callable(
                    getattr(built_object.model, 'predict')):
                self.keras_model = built_object.model
            else:
                logger.error(f"DeepFace.build_model for '{self.model_name}' returned an unexpected object type.")
                self.keras_model = None
                return

            if self.keras_model is None:
                logger.error(f"Could not obtain a valid Keras model for {self.model_name}.")
                return

            self.keras_model.trainable = False

            if self.keras_model.input_shape and len(self.keras_model.input_shape) >= 3:
                self.target_size = self.keras_model.input_shape[1:3]
            else:
                logger.error(
                    f"Could not determine target_size from Keras model input_shape: {self.keras_model.input_shape}")
                self.keras_model = None
                return

            dummy_input_0_1 = np.zeros((1, *self.target_size, 3), dtype=np.float32)
            dummy_input_normalized = self._normalize_for_model(dummy_input_0_1[0])
            _ = self.keras_model(np.expand_dims(dummy_input_normalized, axis=0).astype(np.float32), training=False)

            logger.info(
                f"Keras model for '{self.model_name}' loaded. Target input (H,W): {self.target_size}. Init time: {time.time() - init_start_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load Keras model '{self.model_name}' or perform dummy prediction: {e}",
                         exc_info=True)
            self.keras_model = None

    def _normalize_for_model(self, face_img_array_0_1):
        return (face_img_array_0_1 * 2.0) - 1.0

    def _get_preprocessed_face_and_embedding(self, image_path):
        if self.keras_model is None or self.target_size is None:
            logger.error("SHAP: Keras embedding model or target_size not available for preprocessing.")
            return None, None, None

        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist for SHAP preprocessing: {image_path}")
            return None, None, None

        preprocess_start_time = time.time()
        logger.debug(f"SHAP: Starting preprocessing for {image_path}")
        try:
            # Using 'retinaface' detector for a potential speedup over 'mtcnn'
            img_data_list = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='retinaface',  # Changed from 'mtcnn'
                enforce_detection=True,
                align=True
            )

            if not img_data_list or 'face' not in img_data_list[0]:
                logger.warning(f"SHAP: No face detected or extracted from {image_path} using retinaface.")
                return None, None, None

            extracted_face_0_1 = img_data_list[0]['face']

            pil_image = Image.fromarray((extracted_face_0_1 * 255).astype(np.uint8))
            pil_image_resized = pil_image.resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR)
            original_face_0_1_resized = np.array(pil_image_resized) / 255.0

            if original_face_0_1_resized.ndim == 2:
                original_face_0_1_resized = np.stack((original_face_0_1_resized,) * 3, axis=-1)

            normalized_face_for_model = self._normalize_for_model(original_face_0_1_resized)
            model_input_face = np.expand_dims(normalized_face_for_model, axis=0).astype(np.float32)

            embedding_batch = self.keras_model(tf.cast(model_input_face, tf.float32), training=False)
            embedding = embedding_batch.numpy()[0]

            logger.debug(
                f"SHAP: Preprocessing for {os.path.basename(image_path)} took {time.time() - preprocess_start_time:.2f}s. Embedding shape: {embedding.shape}")
            return original_face_0_1_resized.astype(np.float32), model_input_face.astype(np.float32), embedding.astype(
                np.float32)

        except Exception as e:
            if "Face could not be detected" in str(e) or "No face detected" in str(e):
                logger.warning(f"SHAP: No face detected in {image_path}. Error: {e}")
            else:
                logger.error(f"Error during SHAP preprocessing for {image_path}: {e}", exc_info=True)
            return None, None, None

    def explain_similarity(self, image_query_path, image_retrieved_path):
        if self.keras_model is None:
            logger.error("SHAP: Keras embedding model not loaded, cannot explain similarity.")
            return None, None

        overall_start_time = time.time()
        logger.info(
            f"SHAP: Explaining similarity between {os.path.basename(image_query_path)} and {os.path.basename(image_retrieved_path)}")

        orig_face_q_0_1_resized, model_input_q, emb_query = self._get_preprocessed_face_and_embedding(image_query_path)
        if emb_query is None: return None, None

        orig_face_r_0_1_resized, model_input_r, emb_retrieved = self._get_preprocessed_face_and_embedding(
            image_retrieved_path)
        if emb_retrieved is None: return None, None

        logger.info("SHAP: Preprocessing and embedding complete. Defining SHAP models.")

        fixed_emb_retrieved_tf = tf.constant(emb_retrieved.reshape(1, -1), dtype=tf.float32)
        fixed_emb_query_tf = tf.constant(emb_query.reshape(1, -1), dtype=tf.float32)

        model_input_shape_no_batch = self.keras_model.input_shape[1:]

        image_input_for_q_model = tf.keras.Input(shape=model_input_shape_no_batch, dtype=tf.float32,
                                                 name="image_input_q_shap")
        embedding_from_input_q = self.keras_model(image_input_for_q_model, training=False)
        similarity_score_q = tf.reduce_sum(embedding_from_input_q * fixed_emb_retrieved_tf, axis=1)
        similarity_output_q = tf.reshape(similarity_score_q, [-1, 1], name="similarity_output_q")
        shap_keras_model_q = tf.keras.Model(inputs=image_input_for_q_model, outputs=similarity_output_q,
                                            name="ShapQuerySimilarityModel")

        image_input_for_r_model = tf.keras.Input(shape=model_input_shape_no_batch, dtype=tf.float32,
                                                 name="image_input_r_shap")
        embedding_from_input_r = self.keras_model(image_input_for_r_model, training=False)
        similarity_score_r = tf.reduce_sum(embedding_from_input_r * fixed_emb_query_tf, axis=1)
        similarity_output_r = tf.reshape(similarity_score_r, [-1, 1], name="similarity_output_r")
        shap_keras_model_r = tf.keras.Model(inputs=image_input_for_r_model, outputs=similarity_output_r,
                                            name="ShapRetrievedSimilarityModel")

        black_img_0_1_resized = np.zeros_like(orig_face_q_0_1_resized, dtype=np.float32)
        normalized_black_img = self._normalize_for_model(black_img_0_1_resized)
        background_samples_np = np.expand_dims(normalized_black_img, axis=0).astype(np.float32)

        shap_values_q_arr, shap_values_r_arr = None, None
        try:
            shap_q_start_time = time.time()
            logger.info("SHAP: Computing explanations for query image...")
            explainer_q = shap.GradientExplainer(shap_keras_model_q, background_samples_np)
            # GradientExplainer output for a model with 1 output is typically a list [array_of_shape(N, H, W, C_in)]
            # If the model's output was (N, M_out), it would be a list of M_out arrays.
            # The log showed (1, H, W, C, 1), meaning the array itself has the output dim.
            raw_shap_q = explainer_q.shap_values(model_input_q.astype(np.float32))

            # raw_shap_q is expected to be a list of arrays, one per model output. Our model has 1 output.
            # The array inside the list has shape (batch_size, H, W, C_input, C_output=1)
            if isinstance(raw_shap_q, list) and len(raw_shap_q) > 0:
                shap_5d_q = raw_shap_q[0]  # Shape: (1, H, W, C_in, 1)
                if shap_5d_q.ndim == 5 and shap_5d_q.shape[0] == 1 and shap_5d_q.shape[-1] == 1:
                    shap_values_q_arr = np.squeeze(shap_5d_q, axis=(0, -1))  # Squeeze batch and output_channel dims
                else:  # Fallback if shape is not as expected
                    logger.warning(f"Unexpected SHAP value array shape for query: {shap_5d_q.shape}")
                    shap_values_q_arr = np.squeeze(shap_5d_q)  # Generic squeeze
            else:  # Should not happen if explainer works as expected
                logger.error(f"raw_shap_q is not a list or is empty: {type(raw_shap_q)}")
                shap_values_q_arr = np.squeeze(raw_shap_q) if raw_shap_q is not None else None

            logger.info(
                f"SHAP: Query SHAP values computed. Time: {time.time() - shap_q_start_time:.2f}s. Shape: {shap_values_q_arr.shape if shap_values_q_arr is not None else 'None'}")

            shap_r_start_time = time.time()
            logger.info("SHAP: Computing explanations for retrieved image...")
            explainer_r = shap.GradientExplainer(shap_keras_model_r, background_samples_np)
            raw_shap_r = explainer_r.shap_values(model_input_r.astype(np.float32))

            if isinstance(raw_shap_r, list) and len(raw_shap_r) > 0:
                shap_5d_r = raw_shap_r[0]
                if shap_5d_r.ndim == 5 and shap_5d_r.shape[0] == 1 and shap_5d_r.shape[-1] == 1:
                    shap_values_r_arr = np.squeeze(shap_5d_r, axis=(0, -1))
                else:
                    logger.warning(f"Unexpected SHAP value array shape for retrieved: {shap_5d_r.shape}")
                    shap_values_r_arr = np.squeeze(shap_5d_r)
            else:
                logger.error(f"raw_shap_r is not a list or is empty: {type(raw_shap_r)}")
                shap_values_r_arr = np.squeeze(raw_shap_r) if raw_shap_r is not None else None

            logger.info(
                f"SHAP: Retrieved SHAP values computed. Time: {time.time() - shap_r_start_time:.2f}s. Shape: {shap_values_r_arr.shape if shap_values_r_arr is not None else 'None'}")

        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}", exc_info=True)
            return None, None

        plot_start_time = time.time()
        logger.info("SHAP: Plotting heatmaps...")
        heatmap_q_path = self.plot_and_save_shap_heatmap(orig_face_q_0_1_resized.astype(np.float32), shap_values_q_arr,
                                                         "query_explanation")
        heatmap_r_path = self.plot_and_save_shap_heatmap(orig_face_r_0_1_resized.astype(np.float32), shap_values_r_arr,
                                                         "retrieved_explanation")
        logger.info(f"SHAP: Plotting heatmaps took {time.time() - plot_start_time:.2f}s.")

        if not heatmap_q_path or not heatmap_r_path:
            logger.error("Failed to generate or save one or both SHAP heatmaps.")
            return None, None

        logger.info(f"SHAP: Explanation generation complete. Total time: {time.time() - overall_start_time:.2f}s")
        return heatmap_q_path, heatmap_r_path

    def plot_and_save_shap_heatmap(self, original_image_0_1_resized, shap_values_arr, base_filename_prefix):
        if original_image_0_1_resized is None or shap_values_arr is None:
            logger.warning(f"Cannot plot: Missing image or SHAP values for {base_filename_prefix}.")
            return None

        # After squeezing, shap_values_arr should be (H, W, C_input) e.g. (112, 112, 3)
        if not (shap_values_arr.ndim == 3 and shap_values_arr.shape == original_image_0_1_resized.shape):
            logger.error(
                f"SHAP values for {base_filename_prefix} have incorrect shape {shap_values_arr.shape}. "
                f"Expected 3D shape matching image {original_image_0_1_resized.shape}."
            )
            return None

        try:
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{base_filename_prefix}_{unique_id}.png"
            output_path = os.path.join(config.EXPLANATIONS_DIR_ABS_PATH, filename)

            # shap.image_plot expects SHAP values as (batch, H, W, C) or (batch, H, W)
            # and pixel_values (original images) also (batch, H, W, C)
            # Our shap_values_arr is now (H,W,C), original_image_0_1_resized is (H,W,C)
            shap_values_for_plot = np.expand_dims(shap_values_arr.astype(np.float64), axis=0)  # Add batch dim
            pixel_values_for_plot = np.expand_dims(original_image_0_1_resized.astype(np.float64),
                                                   axis=0)  # Add batch dim

            # Sum SHAP values over color channels for a single heatmap overlay
            shap_values_overlay = np.sum(shap_values_for_plot, axis=-1, keepdims=True)

            if np.isnan(shap_values_overlay).any() or np.isinf(shap_values_overlay).any():
                logger.warning(f"Clamping NaN/Inf in SHAP values for {base_filename_prefix} plot.")
                shap_values_overlay = np.nan_to_num(shap_values_overlay, nan=0.0, posinf=1e9, neginf=-1e9)

            fig = plt.figure()
            shap.image_plot(
                shap_values=shap_values_overlay,
                pixel_values=pixel_values_for_plot,
                show=False
            )
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"SHAP heatmap saved: {output_path}")
            return f"{config.EXPLANATIONS_URL_PREFIX}/{filename}"

        except Exception as e:
            logger.error(f"Error plotting/saving SHAP heatmap for {base_filename_prefix}: {e}", exc_info=True)
            if 'output_path' in locals() and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception as e_rem:
                    logger.error(f"Could not remove partial heatmap {output_path}: {e_rem}")
            return None