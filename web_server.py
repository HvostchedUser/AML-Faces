# web_server.py
import os
import uuid
from flask import Flask, request, jsonify, render_template, \
    send_from_directory  # Ensure send_from_directory is imported
from werkzeug.utils import secure_filename

# Important: Ensure fiw_family_searcher is in PYTHONPATH or project root is CWD
from fiw_family_searcher.search.query_processor import QueryProcessor
from fiw_family_searcher import config  # To access model paths for checks
from fiw_family_searcher.utils.helpers import setup_logger

logger = setup_logger(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'interface/uploads'  # Temporary storage for uploaded photos
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='interface', static_folder='interface/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

# --- Ensure upload folder exists ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global QueryProcessor instance ---
# Check if models and necessary files exist before initializing
query_processor_instance = None


def check_required_files():
    required_files = [
        config.FAMILY_CLASSIFIER_PATH,
        config.RELATIONSHIP_CLASSIFIER_PATH,
        os.path.join(config.EMBEDDINGS_DIR, "person_avg_face_embeddings.pkl"),
        config.FACE_INDEX_PATH, config.NAME_INDEX_PATH,
        config.FACE_ID_MAP_PATH, config.NAME_ID_MAP_PATH
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"CRITICAL: QueryProcessor cannot start. Missing required files: {missing_files}")
        logger.error("Please run 'python main.py full_setup_train' first.")
        return False
    return True


if check_required_files():
    try:
        query_processor_instance = QueryProcessor()
        logger.info("QueryProcessor initialized successfully for web server.")
    except Exception as e:
        logger.error(f"Failed to initialize QueryProcessor: {e}")
        query_processor_instance = None  # Explicitly set to None on failure
else:
    logger.warning("QueryProcessor not initialized due to missing files.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


# NEW ROUTE TO SERVE MEMBER IMAGES
@app.route('/member_images/<path:subpath>')
def serve_member_image(subpath):
    # subpath is expected to be like "FIDxxxx/MIDy/photoname.jpg"
    # config.FIDS_FULL_DIR is the absolute base path to the FIDs directory
    # Basic security check to prevent accessing files outside the intended directory
    if '..' in subpath or subpath.startswith('/'):
        logger.warning(f"Attempt to access invalid image path: {subpath}")
        return "Invalid path", 400

    logger.debug(f"Attempting to serve image from directory: {config.FIDS_FULL_DIR}, subpath: {subpath}")
    try:
        # send_from_directory expects the directory and the filename (subpath) within that directory
        return send_from_directory(config.FIDS_FULL_DIR, subpath, as_attachment=False)
    except FileNotFoundError:
        logger.error(f"Image not found at: {os.path.join(config.FIDS_FULL_DIR, subpath)}")
        # Optionally, return a placeholder image or just 404
        return "Image not found", 404
    except Exception as e:
        logger.error(f"Error serving image {subpath}: {e}")
        return "Error serving image", 500


@app.route('/query', methods=['POST'])
def handle_query():
    if query_processor_instance is None:
        return jsonify(
            {"error": "Backend processor is not ready. System setup might be incomplete. Check server logs."}), 503

    if 'photo' not in request.files:
        return jsonify({"error": "No photo part in the request"}), 400

    file = request.files['photo']
    user_name = request.form.get('name', None)

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        # Create a unique filename to avoid conflicts and for temp storage
        unique_suffix = uuid.uuid4().hex
        temp_filename = f"{os.path.splitext(original_filename)[0]}_{unique_suffix}{os.path.splitext(original_filename)[1]}"
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        try:
            file.save(photo_path)
            logger.info(f"Photo saved to {photo_path}")

            # The QueryProcessor expects paths relative to its execution or absolute paths
            # Ensure photo_path is absolute if QueryProcessor needs it
            absolute_photo_path = os.path.abspath(photo_path)

            logger.info(f"Processing query for photo: {absolute_photo_path}, name: {user_name}")
            results = query_processor_instance.find_family_and_relations(absolute_photo_path, user_name)
            logger.info("Query processing complete.")

            return jsonify(results)

        except Exception as e:
            logger.error(f"Error during query processing: {e}", exc_info=True)
            return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500
        finally:
            # --- Clean up the uploaded file ---
            if os.path.exists(photo_path):
                try:
                    os.remove(photo_path)
                    logger.info(f"Temporary photo {photo_path} removed.")
                except Exception as e_remove:
                    logger.error(f"Error removing temporary photo {photo_path}: {e_remove}")
    else:
        return jsonify({"error": "File type not allowed"}), 400


if __name__ == '__main__':
    if query_processor_instance is None:
        print("********************************************************************************")
        print("WARNING: QueryProcessor could not be initialized. The web server will run,")
        print("but queries will fail. Please check the logs for missing files or errors,")
        print("and ensure 'python main.py full_setup_train' has been completed successfully.")
        print("********************************************************************************")
    app.run(debug=True, host='0.0.0.0', port=5000)