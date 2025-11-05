# resume_processor.py
import os
import re
import json
import time
from spellchecker import SpellChecker
from docx import Document
import pdfplumber

# New OpenAI v1 client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Regex constants
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}")
URL_RE = re.compile(r'https?://[^\s]+', re.IGNORECASE)

ACTION_VERBS = [
    "led", "developed", "implemented", "designed", "built", "optimized",
    "launched", "improved", "reduced", "increased", "achieved", "managed",
    "automated", "scaled", "deployed", "created", "engineered", "trained"
]

PORTFOLIO_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "linkedin.com",
                   "kaggle.com", "huggingface.co", "behance.net", "dribbble.com",
                   "medium.com", "substack.com", "notion.so", "read.cv")

# ---------- text extraction ----------
def extract_text_from_pdf(path):
    texts = []
    pages = 0
    try:
        with pdfplumber.open(path) as pdf:
            pages = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
    except Exception:
        pass
    return "\n".join(texts), pages

def extract_text_from_docx(path):
    paragraphs = []
    pages = None
    try:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
    except Exception:
        pass
    return "\n".join(paragraphs), pages

def extract_text(file_path):
    lower = (file_path or "").lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    if lower.endswith(".docx"):
        return extract_text_from_docx(file_path)
    try:
        with open(file_path, "rb") as f:
            raw = f.read().decode("utf-8", errors="ignore")
            return raw, None
    except Exception:
        return "", None

# ---------- small extractors ----------
def extract_skills(text, job_keywords=None):
    common_skills = ["python","java","c++","sql","aws","docker","tensorflow","pytorch","nlp","mlops","ci/cd","react","laravel"]
    skills_found = [kw for kw in (job_keywords or []) + common_skills if kw.lower() in (text or "").lower()]
    # unique preserve order
    seen = set()
    out = []
    for s in skills_found:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def extract_projects(text):
    project_lines = []
    lines = (text or "").splitlines()
    for i, l in enumerate(lines):
        if 'project' in l.lower():
            snippet = " ".join(lines[i+1:i+4])
            if snippet.strip():
                project_lines.append(snippet.strip())
    return project_lines

def extract_languages(text):
    langs = []
    for l in ["English","Spanish","French","German","Chinese","Java","Python","C++","SQL","PHP"]:
        if l.lower() in (text or "").lower():
            langs.append(l)
    return langs

def extract_links(text):
    return URL_RE.findall(text or "")

def extract_job_description(job_keywords):
    if not job_keywords:
        return "No job description available."
    return "Role expects: " + ", ".join(job_keywords)

def extract_resume_job_description(text):
    """
    Heuristic extraction of a job/role/summary/objective from the resume text.
    Looks for headings like 'Objective', 'Summary', 'Professional Summary', 'Career Objective'.
    Returns a short snippet (1–3 lines).
    """
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    heads = ("objective", "summary", "professional summary", "career objective", "profile")
    idxs = [i for i, l in enumerate(lines) if any(h in l.lower() for h in heads)]
    if idxs:
        i = idxs[0]
        snippet = " ".join(lines[i+1:i+4])
        return snippet[:600]
    # fallback: first 3 lines as pseudo summary
    return " ".join(lines[:3])[:600]

# ---------- heuristic (feature score only) ----------
def heuristic_feature_score(text):
    """
    Small ATS-like feature score (0-100). Intentionally simple:
    - keyword presence boosts score
    - spelling mistakes penalize
    - contact presence small boost
    """
    t = (text or "").lower()
    keywords = ["python","aws","docker","sql","tensorflow","pytorch","nlp","mlops"]
    score = 0
    for kw in keywords:
        if kw in t:
            score += 12
    score = min(score, 80)
    spell = SpellChecker()
    tokens = re.findall(r"[A-Za-z']{2,}", text or "")
    miss = spell.unknown([w for w in tokens if w.isalpha()])
    penalty = min(len(miss), 30)
    score = max(0, score - penalty)
    contact_ok = bool(EMAIL_RE.search(text or "") or PHONE_RE.search(text or ""))
    if contact_ok:
        score += 5
    return max(0, min(100, int(round(score))))

# ---------- robust JSON extraction ----------
def _extract_json(candidate_text):
    """
    Try to find JSON object within text and parse it. Try simple repairs.
    Returns (dict or None, raw_json_text)
    """
    txt = (candidate_text or "").strip()
    # find first {...}
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    target = m.group(0) if m else txt

    attempts = [target]
    # replace smart quotes
    attempts.append(target.replace("“", '"').replace("”", '"').replace("’", "'"))
    # remove trailing commas before } or ]
    attempts.append(re.sub(r",\s*([\]}])", r"\1", attempts[-1]))
    for c in attempts:
        try:
            return json.loads(c), c
        except Exception:
            # try naive key quoting + single->double
            try:
                repaired = re.sub(r"(\w+)\s*:", r'"\1":', c)
                repaired = repaired.replace("'", '"')
                return json.loads(repaired), repaired
            except Exception:
                continue
    return None, txt

# ---------- OpenAI-based analysis (v1 client) ----------
def ai_analyze_with_openai(text, job_title="", job_keywords=None, model=None, openai_api_key=None, resume_job_desc=""):
    """
    Use OpenAI v1 client (OpenAI()) to request structured JSON output.
    Returns a dict (either parsed ai JSON or fallback error structure).
    """
    if OpenAI is None:
        return {
            "score": None,
            "selected": False,
            "resume_category": "openai_sdk_missing",
            "job_match_score": 0.0,
            "reasons": ["OpenAI client not installed."],
            "suggestions": [],
            "feature_scores": {},
            "detailed_scores": {},
            "report": "OpenAI client not available on server."
        }

    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
    else:
        client = OpenAI()  # will use env OPENAI_API_KEY if present

    model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    system = (
        "You are an expert technical recruiter and resume coach. Provide a single JSON object only."
    )

    prompt = (
        "Analyze the resume below and RETURN ONLY A SINGLE PARSABLE JSON OBJECT (no extra commentary). "
        "The JSON MUST contain keys: "
        "score (float 0-10), selected (boolean), resume_category (string), "
        "job_match_score (float 0-10), reasons (array of short strings), suggestions (array of short strings), "
        "feature_scores (object numeric), detailed_scores (skills, projects, experience, languages, certifications each {score, review}), "
        "report (short 1-3 sentence summary). "
        "Additionally, INCLUDE optional keys: why_selected (array of strings), why_not_selected (array of strings), "
        "conclusion (string), resume_2025_tips (array of short tips). Keep arrays short.\n\n"
        f"Job Title: {job_title}\n"
        f"Job Keywords: {job_keywords}\n"
        f"Job Description (Provided): {extract_job_description(job_keywords)}\n"
        f"Job/Objective Extracted from Resume: {resume_job_desc}\n\n"
        "Resume text:\n\n" + (text or "")[:12000]
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=900
        )
        # Extract content (v1: resp.choices[0].message.content)
        out = ""
        try:
            out = resp.choices[0].message.content
        except Exception:
            out = str(resp)

        parsed, raw = _extract_json(out)
        if parsed is None:
            return {
                "score": None,
                "selected": False,
                "resume_category": "parse_failed",
                "job_match_score": 0.0,
                "reasons": ["AI returned unparsable JSON."],
                "suggestions": [],
                "feature_scores": {},
                "detailed_scores": {},
                "report": out
            }

        # Normalize parsed fields and ensure keys exist
        parsed_out = {}
        parsed_out["score"] = float(parsed.get("score")) if parsed.get("score") not in (None, "") else None
        parsed_out["selected"] = bool(parsed.get("selected", parsed_out["score"] is not None and parsed_out["score"] >= 7.5))
        parsed_out["resume_category"] = parsed.get("resume_category") or parsed.get("category") or ""
        parsed_out["job_match_score"] = float(parsed.get("job_match_score", parsed.get("job_match", 0.0) or 0.0))
        parsed_out["reasons"] = parsed.get("reasons", []) or []
        parsed_out["suggestions"] = parsed.get("suggestions", []) or []
        parsed_out["feature_scores"] = parsed.get("feature_scores", {}) or {}
        parsed_out["detailed_scores"] = parsed.get("detailed_scores", {}) or {}
        parsed_out["report"] = parsed.get("report", "") or parsed.get("summary", "")
        # Optional extras
        parsed_out["why_selected"] = parsed.get("why_selected", []) or []
        parsed_out["why_not_selected"] = parsed.get("why_not_selected", []) or []
        parsed_out["conclusion"] = parsed.get("conclusion", "")
        parsed_out["resume_2025_tips"] = parsed.get("resume_2025_tips", []) or []
        parsed_out["_raw_model_output"] = raw
        return parsed_out

    except Exception as e:
        return {
            "score": None,
            "selected": False,
            "resume_category": "ai_error",
            "job_match_score": 0.0,
            "reasons": [f"OpenAI error: {e}"],
            "suggestions": [],
            "feature_scores": {},
            "detailed_scores": {},
            "report": f"OpenAI error: {e}",
            "why_selected": [],
            "why_not_selected": [],
            "conclusion": "",
            "resume_2025_tips": []
        }

# ---------- convert ai_data to Markdown for UI ----------
def generate_dynamic_ai_report(ai_data, feature_score, job_title=""):
    """
    Convert ai_data (structured) into pretty Markdown with emojis.
    Also includes thresholded verdict, why selected/not selected, and 2025 guidance.
    Requires that caller populate ai_data['_final_score'] and ai_data['_selected_threshold'] (we add them in process_resume).
    """
    lines = []
    lines.append(f"📑 **AI Resume Analysis — {job_title or 'Candidate'}**\n")
    ai_score = ai_data.get("score")
    if ai_score is not None:
        lines.append(f"🔹 **AI Score:** {ai_score}/10")
    else:
        lines.append("🔹 **AI Score:** N/A")

    final_score = ai_data.get("_final_score")
    threshold = ai_data.get("_selected_threshold", 7.5)
    if final_score is not None:
        lines.append(f"🔹 **Final Score (used for decision):** {final_score}/10 (threshold: {threshold})")

    lines.append(f"🔹 **Feature Score (heuristic):** {feature_score}/100")

    selected = bool(ai_data.get("_selected", False))
    lines.append(f"🔹 **Selection Status:** {'Selected ✅' if selected else 'Not Selected ❌'}")
    verdict_label = "✅ **Accepted / Excellent Resume**" if selected else "❌ **Rejected / Bad Resume**"
    lines.append(f"\n### {verdict_label}\n")

    cat = ai_data.get("resume_category") or "Unknown"
    lines.append(f"🔹 **Category:** {cat}\n")

    # Why selected / not selected
    why_sel = ai_data.get("why_selected") or []
    why_not = ai_data.get("why_not_selected") or []
    if selected:
        lines.append("#### 🏆 Why this resume is selected?")
        if why_sel:
            for r in why_sel[:8]:
                lines.append(f"- {r}")
        else:
            # fallback: infer from reasons/suggestions
            jm = ai_data.get("job_match_score")
            if jm is not None:
                lines.append(f"- Strong job match score ({jm}/10).")
            lines.append("- Adequate alignment with role keywords and experience.")
    else:
        lines.append("#### ⚠️ Why this resume is not selected?")
        if why_not:
            for r in why_not[:10]:
                lines.append(f"- {r}")
        else:
            # fallback: use reasons
            reasons = ai_data.get("reasons") or []
            if reasons:
                for r in reasons[:10]:
                    lines.append(f"- {r}")
            else:
                lines.append("- Insufficient evidence of role alignment or missing core requirements.")

    rep = ai_data.get("report") or ""
    if rep:
        lines.append(f"\n📝 **Summary:**\n{rep}\n")

    # Suggestions
    suggestions = ai_data.get("suggestions") or []
    if suggestions:
        lines.append("\n💡 **Top Suggestions to Improve:**")
        for s in suggestions[:12]:
            lines.append(f"- {s}")

    # AI feature scores (if model provided any)
    fs = ai_data.get("feature_scores", {}) or {}
    if fs:
        lines.append("\n📊 **Feature Scores (AI):**")
        for k, v in fs.items():
            lines.append(f"- {k}: {v}")

    # Detailed section scores
    ds = ai_data.get("detailed_scores", {}) or {}
    if ds:
        lines.append("\n🔍 **Detailed Section Scores:**")
        for k, v in ds.items():
            if isinstance(v, dict):
                lines.append(f"- {k}: {v.get('score', '')} — {v.get('review','')[:200]}")
            else:
                lines.append(f"- {k}: {v}")

    # Conclusion
    concl = ai_data.get("conclusion", "")
    if concl or True:
        lines.append("\n**📌 AI Conclusion:**")
        if concl:
            lines.append(concl)
        else:
            lines.append("Overall, the resume shows potential. Focus on quantifying impact, aligning keywords to the role, and showcasing links to relevant work to cross the selection threshold.")

    # 2025 guidance
    tips_2025 = ai_data.get("resume_2025_tips") or []
    lines.append("\n### 🗓️ 2025 Resume Guidance")
    if tips_2025:
        for t in tips_2025[:10]:
            lines.append(f"- {t}")
    else:
        lines.append("- Use a clean 1–2 page layout with strong section headers and plenty of white space.")
        lines.append("- Quantify impact with metrics (e.g., latency ↓25%, revenue +$1.2M, accuracy +4.3pp).")
        lines.append("- Align keywords with the job description; mirror terminology from the posting.")
        lines.append("- Link to portfolio/GitHub/LinkedIn; add 1–3 flagship projects with brief tech + outcome.")
        lines.append("- Put the strongest content in the top third of page 1; recruiters skim first.")
        lines.append("- For ML roles: include model types, data sizes, MLOps stack, deployment context, and monitoring.")

    return "\n".join(lines)

# ---------- main entry ----------
def process_resume(file_path, job_title="", job_keywords=None, use_openai=False, openai_api_key=None, openai_model=None):
    """
    Returns an analysis dict:
      - final_score (0-10 float)
      - feature_scores (dict)  # heuristic-derived small values or empty
      - feature_score (0-100 int)
      - selected (bool)  # threshold at 7.5
      - ai_report (Markdown string)
      - ai_data (structured dict from model or fallback)
      - skills, projects, languages, links, pages
      - resume_job_description (string extracted from resume)
    """
    text, pages = extract_text(file_path)

    # heuristic feature score
    feature_score = heuristic_feature_score(text)

    # links metrics for feature table (0-10)
    links = extract_links(text)
    links_count = len(links)
    # simple score: 0 if none, else min(10, 3*count) with bonus if portfolio hosts present
    portfolio_hits = sum(1 for u in links if any(h in u.lower() for h in PORTFOLIO_HOSTS))
    links_score = 0
    if links_count > 0:
        links_score = min(10, links_count * 3 + (2 if portfolio_hits > 0 else 0))

    # extract a job/objective-like description from resume
    resume_job_desc = extract_resume_job_description(text)

    ai_data = {
        "score": None,
        "selected": False,
        "resume_category": None,
        "job_match_score": 0.0,
        "reasons": [],
        "suggestions": [],
        "feature_scores": {},
        "detailed_scores": {},
        "report": "",
        "why_selected": [],
        "why_not_selected": [],
        "conclusion": "",
        "resume_2025_tips": []
    }

    if use_openai:
        ai_data = ai_analyze_with_openai(
            text=text,
            job_title=job_title,
            job_keywords=job_keywords,
            model=openai_model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            openai_api_key=openai_api_key,
            resume_job_desc=resume_job_desc
        )

    # determine final_score: prefer AI score, otherwise scale heuristic
    final_score = None
    if ai_data.get("score") is not None:
        try:
            final_score = round(float(ai_data.get("score")), 1)
        except Exception:
            final_score = None

    if final_score is None:
        final_score = round((feature_score / 100.0) * 10, 1)

    # Enforce selection threshold at 7.5 exactly as requested
    selected = final_score >= 7.5

    # ensure feature_scores always present (merge heuristic small metrics)
    feature_scores = (ai_data.get("feature_scores") or {}).copy()
    # add quick heuristic-derived small indicators to feature_scores to show up in UI (scale 0-10)
    feature_scores.setdefault("heuristic_feature_score", round(feature_score / 10.0, 2))
    # links-related feature metrics (so they appear in the Feature Scores table/graph)
    feature_scores.setdefault("links_count", links_count)
    feature_scores.setdefault("links_score_0to10", links_score)
    feature_scores.setdefault("portfolio_links", portfolio_hits)

    ai_data["feature_scores"] = feature_scores

    # Attach final score & threshold for report rendering
    ai_data["_final_score"] = final_score
    ai_data["_selected_threshold"] = 7.5
    ai_data["_selected"] = selected

    ai_report_text = generate_dynamic_ai_report(ai_data, feature_score, job_title=job_title)

    analysis = {
        "final_score": final_score,
        "feature_score": feature_score,
        "feature_scores": feature_scores,
        "selected": selected,
        "ai_report": ai_report_text,
        "ai_data": ai_data,
        "skills": extract_skills(text, job_keywords),
        "projects": extract_projects(text),
        "languages": extract_languages(text),
        "links": links,
        "pages": pages,
        "resume_job_description": resume_job_desc
    }

    return analysis

# quick manual test helper if run directly
if __name__ == "__main__":
    TEST = os.getenv("TEST_FILE")
    KEY = os.getenv("OPENAI_API_KEY")
    if not TEST:
        print("Set TEST_FILE to a resume path to test.")
    else:
        print("Processing", TEST)
        out = process_resume(TEST, job_title="AI Engineer", job_keywords=["python","aws"], use_openai=bool(KEY), openai_api_key=KEY)
        print(json.dumps(out, indent=2, default=str))
