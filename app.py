from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
from ring_visual_search import RingVisualSearch

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

Path("uploads").mkdir(exist_ok=True)
Path("catalog").mkdir(exist_ok=True)

engine = RingVisualSearch()
if not engine._load_features_db():
    engine.build_catalog_index()

ALLOWED = {"jpg","jpeg","png","webp","bmp"}

def allowed(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/catalog-info")
def catalog_info():
    imgs = [p for p in Path("catalog").rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp"}]
    return jsonify({"total": len(imgs), "indexed": len(engine.features)})

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not allowed(f.filename):
        return jsonify({"error": "Invalid file"}), 400

    top_k     = int(request.form.get("top_k", 10))
    threshold = float(request.form.get("threshold", 85))

    filename = secure_filename(f.filename)
    path     = os.path.join("uploads", filename)
    f.save(path)

    try:
        results = engine.search(path, top_k=top_k, similarity_threshold=threshold)
        matches = [{
            "index":      r["index"],
            "similarity": round(r["match_percentage"], 1),
            "image_url":  f"/catalog-image/{Path(r['image_path']).name}",
            "filename":   r["metadata"]["filename"],
            "variants":   r.get("duplicate_count", 1),
        } for r in results]

        return jsonify({
            "query_image": f"/uploads/{filename}",
            "matches":     matches,
            "total":       len(engine.features),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/customize/<int:idx>")
def customize(idx):
    return jsonify(engine.get_customization_options(idx))

@app.route("/rebuild", methods=["POST"])
def rebuild():
    try:
        engine.build_catalog_index()
        return jsonify({"ok": True, "indexed": len(engine.features)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory("uploads", filename)

@app.route("/catalog-image/<filename>")
def serve_catalog(filename):
    return send_from_directory("catalog", filename)

if __name__ == "__main__":
    print("Open http://localhost:5000")
    app.run(debug=True, port=5000)
