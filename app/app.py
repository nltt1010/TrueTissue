import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

from services.tissue_analyzer import TissueAnalyzer

# Initialize Flask app with structured static and template directories
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

# Ensure upload and output directories exist
UPLOADS_DIR = Path(__path__[0]) if hasattr(app, '__path__') else Path(__file__).resolve().parent / "static" / "uploads"
OUTPUTS_DIR = Path(__file__).resolve().parent / "static" / "outputs"
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

# Initialize AI Engine
analyzer = TissueAnalyzer()


@app.route("/")
def index():
    """Renders the primary Diagnostic Staining & Abnormality Analysis Workspace."""
    return render_template("index.html", active_page="analyze")


@app.route("/compare")
def compare():
    """Renders the Multi-Model & Pipeline Step Comparison Suite."""
    return render_template("compare.html", active_page="compare")


@app.route("/api/checkpoints", methods=["GET"])
def get_checkpoints():
    """Returns all discovered checkpoint filenames for basemodel, mymodel, and classifier."""
    try:
        checkpoints = analyzer.get_available_checkpoints()
        return jsonify({"status": "success", "data": checkpoints}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns available built-in demo test tissue samples for instant testing."""
    try:
        samples = analyzer.get_demo_samples()
        return jsonify({"status": "success", "data": samples}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def _get_input_image_path(req):
    """Helper to resolve whether the user uploaded a file or selected a built-in demo sample."""
    if "file" in req.files and req.files["file"].filename != "":
        file = req.files["file"]
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            raise ValueError("Unsupported file format. Please upload PNG, JPG, or TIF.")
        
        filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
        filepath = UPLOADS_DIR / filename
        file.save(str(filepath))
        return str(filepath)
    elif req.form.get("sample_path"):
        sample_path = req.form.get("sample_path")
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Selected sample not found on disk: {sample_path}")
        return sample_path
    else:
        raise ValueError("No image file uploaded or sample selected.")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Executes single-model analysis pipeline:
    Stain (Base or My Model) -> Predict Abnormality -> Circle Tumor Regions -> 4x High-Res Sharpening.
    """
    try:
        image_path = _get_input_image_path(request)
        model_type = request.form.get("model_type", "mymodel")
        stain_ckpt = request.form.get("stain_ckpt") or None
        cls_ckpt = request.form.get("cls_ckpt") or None
        
        results = analyzer.run_sliced_analysis(
            img_input=image_path,
            model_type=model_type,
            stain_ckpt=stain_ckpt,
            cls_ckpt=cls_ckpt,
            output_dir=OUTPUTS_DIR
        )
        return jsonify({"status": "success", "data": results}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def compare_models():
    """
    Executes simultaneous comparative analysis on BOTH Base Model and My Model.
    Returns metrics and visual artifacts for side-by-side & slider evaluation.
    """
    try:
        image_path = _get_input_image_path(request)
        stain_ckpt_base = request.form.get("stain_ckpt_base") or None
        stain_ckpt_my = request.form.get("stain_ckpt_my") or None
        cls_ckpt = request.form.get("cls_ckpt") or None
        
        comparison_results = analyzer.run_sliced_comparison(
            img_input=image_path,
            stain_ckpt_base=stain_ckpt_base,
            stain_ckpt_my=stain_ckpt_my,
            cls_ckpt=cls_ckpt,
            output_dir=OUTPUTS_DIR
        )
        return jsonify({"status": "success", "data": comparison_results}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("=========================================================")
    print("     TrueTissue Clinical AI Diagnostic Web Server        ")
    print("=========================================================")
    print(" -> Starting Flask server on http://localhost:5000       ")
    app.run(host="0.0.0.0", port=5000, debug=True)
