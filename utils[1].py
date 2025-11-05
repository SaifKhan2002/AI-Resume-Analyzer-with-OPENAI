# utils.py
import time

def analyze_resume(text, job_title, job_keywords):
    """
    Enhanced AI-like analysis for resumes in 2025.
    Always returns a structured report (never None).
    """

    strengths, weaknesses = [], []

    # Keyword matching for strengths
    for kw in job_keywords:
        if kw.lower() in text.lower():
            strengths.append(f"Shows knowledge of {kw}.")

    # Weakness detection (missing keywords)
    for kw in job_keywords:
        if kw.lower() not in text.lower():
            weaknesses.append(f"Missing mention of {kw}.")

    # Additional scoring
    skill_coverage = len(strengths) / max(1, len(job_keywords))
    score = round(skill_coverage * 10, 1)

    # Build formatted AI Report
    report = f"""📑 **AI Resume Analysis Report (2025 Standard)**  

🔹 **Target Role**: {job_title}  
🔹 **Keywords Expected**: {", ".join(job_keywords)}  

✅ **Strengths**  
- {chr(10).join(strengths) if strengths else "No significant strengths found."}  

⚠ **Weaknesses**  
- {chr(10).join(weaknesses) if weaknesses else "No major weaknesses found."}  

📊 **Skills Gap Analysis**  
- Coverage: {len(strengths)}/{len(job_keywords)} ({skill_coverage:.0%}) of required skills mentioned.  

🌱 **Career Growth Potential**  
- Resume shows potential to grow into {job_title} role with additional focus on emerging 2025 trends like Generative AI, cloud-native MLOps, and automation.  

💡 **Final Suggestion**  
{"Resume is strong and ready for shortlist consideration." if score >= 7.5 else "Add more relevant projects, highlight quantified achievements, and align skills to 2025 job market trends."}  
"""

    return {
        "ai_report": report,
        "final_score": score,
        "analyzed_at": time.time()
    }
