# app.py
import os
import time
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
from resume_processor import process_resume, extract_text

# Load environment
load_dotenv()

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXT = {".pdf", ".docx"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "resume_db")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db["resumes"]

# OpenAI config (used by resume_processor via arguments)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Sample job postings
JOB_POSTINGS = [
    {
        "title": "AI Engineer",
        "company": "Acme AI Labs",
        "location": "Remote",
        "desc": "Build and productionize ML models, work with Python, PyTorch, MLOps, and cloud infra.",
        "keywords": ["python", "pytorch", "mlops", "docker", "rest", "api", "aws", "nlp", "transformers"]
    },
    {
        "title": "Machine Learning Engineer",
        "company": "NextGen Models",
        "location": "Remote",
        "desc": "Design and deploy ML pipelines; experience in model training, data pipelines and evaluation metrics.",
        "keywords": ["python", "tensorflow", "data pipelines", "sql", "aws", "ci/cd", "docker", "metrics"]
    }
]

def allowed(filename):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXT)

@app.route("/", methods=["GET", "POST"])
def index():
    import random
    job = random.choice(JOB_POSTINGS)
    if request.method == "POST":
        f = request.files.get("resume")
        if not f or f.filename == "":
            flash("No file selected", "danger")
            return redirect(request.url)
        if not allowed(f.filename):
            flash("Invalid file type. Only PDF and DOCX allowed.", "danger")
            return redirect(request.url)

        saved_name = f"{int(time.time())}_{secure_filename(f.filename)}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
        f.save(path)

        text, pages = extract_text(path)

        doc = {
            "filename": f.filename,
            "saved_path": path,
            "uploaded_at": time.time(),
            "job_title": job["title"],
            "job_keywords": job["keywords"],
            "text": text[:20000],
            "pages": pages,
            "processed": False,
            "analysis": None
        }
        res = collection.insert_one(doc)
        flash(f"Uploaded and saved to DB (id: {res.inserted_id}).", "success")
        return redirect(url_for("index"))
    return render_template("index.html", job=job)

@app.route("/analyze_page", methods=["GET"])
def analyze_page():
    count = collection.count_documents({})
    return render_template("analyze_page.html", count=count)

def _process_one_resume(doc):
    try:
        analysis = process_resume(
            file_path=doc["saved_path"],
            job_title=doc.get("job_title", ""),
            job_keywords=doc.get("job_keywords", []),
            use_openai=bool(OPENAI_KEY),
            openai_api_key=OPENAI_KEY,
            openai_model=OPENAI_MODEL
        )
        collection.update_one({"_id": doc["_id"]}, {"$set": {"processed": True, "analysis": analysis, "analyzed_at": time.time()}})
        return {"id": str(doc["_id"]), "status": "ok", "score": analysis.get("final_score")}
    except Exception as e:
        collection.update_one({"_id": doc["_id"]}, {"$set": {"processed": False, "analysis": {"error": str(e)}}})
        return {"id": str(doc["_id"]), "status": "error", "error": str(e)}

@app.route("/analyze", methods=["POST"])
def analyze_all():
    to_process = list(collection.find({"processed": False}))
    results = []
    for doc in to_process:
        results.append(_process_one_resume(doc))
    flash(f"Processed {len(results)} resumes.", "success")
    return redirect(url_for("results"))

@app.route("/results", methods=["GET"])
def results():
    all_docs = list(collection.find({}))
    selected, not_selected = [], []
    for d in all_docs:
        a = d.get("analysis", {}) or {}
        if a and a.get("final_score") is not None:
            if a["final_score"] >= 7.5:
                selected.append(d)
            else:
                not_selected.append(d)
        else:
            not_selected.append(d)
    chart_data = {"selected_count": len(selected), "not_selected_count": len(not_selected)}
    return render_template("results.html", selected=selected, not_selected=not_selected, chart_data=chart_data)

@app.route("/report/<id>", methods=["GET"])
def report(id):
    try:
        doc = collection.find_one({"_id": ObjectId(id)})
    except Exception:
        doc = None
    if not doc:
        flash("Report not found.", "danger")
        return redirect(url_for("results"))
    # ensure analysis keys exist to avoid template errors
    doc_analysis = doc.get("analysis") or {}
    doc_analysis.setdefault("feature_scores", doc_analysis.get("feature_scores", {}))
    doc_analysis.setdefault("ai_report", doc_analysis.get("ai_report", ""))
    doc["analysis"] = doc_analysis
    return render_template("report.html", doc=doc)

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    file = request.files['resume']
    job_title = request.form.get('job_title')

    if not file or file.filename == "":
        return jsonify({"status": "error", "message": "No file provided."}), 400
    if not allowed(file.filename):
        return jsonify({"status": "error", "message": "File type not allowed."}), 400

    save_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{secure_filename(file.filename)}")
    file.save(save_path)

    text, num_pages = extract_text(save_path)

    job_keywords = []
    for j in JOB_POSTINGS:
        if j["title"].lower() == (job_title or "").lower():
            job_keywords = j.get("keywords", [])
            break

    ai_report = process_resume(
        save_path,
        job_title=job_title,
        job_keywords=job_keywords,
        use_openai=bool(OPENAI_KEY),
        openai_api_key=OPENAI_KEY,
        openai_model=OPENAI_MODEL
    )

    doc = {
        "filename": file.filename,
        "saved_path": save_path,
        "uploaded_at": time.time(),
        "job_title": job_title,
        "job_keywords": job_keywords,
        "text": text[:20000],
        "pages": num_pages,
        "processed": True,
        "analysis": ai_report
    }

    resume_id = collection.insert_one(doc).inserted_id
    return jsonify({"status": "success", "resume_id": str(resume_id)})

@app.route("/clear_db", methods=["POST"])
def clear_db():
    collection.delete_many({})
    flash("All resumes deleted from DB.", "warning")
    return redirect(url_for("index"))

@app.route('/reset_resumes')
def reset_resumes():
    db.resumes.delete_many({})
    flash("All resumes have been reset successfully!", "success")
    return redirect(url_for('results'))

if __name__ == "__main__":
    app.run(debug=True)
