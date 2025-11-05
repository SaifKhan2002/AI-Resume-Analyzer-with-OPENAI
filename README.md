# 🧠 AI Resume Analyzer – Flask + OpenAI + MongoDB

An intelligent **AI-powered Resume Analyzer** built with **Flask**, **MongoDB**, and **OpenAI models**, designed to automatically evaluate resumes, match them to job descriptions, and send **selection or improvement feedback emails** to candidates.

---

## 🚀 Features

✅ Upload resumes (PDF/DOCX)
✅ Automatically extract text using AI
✅ Compare skills and match against job descriptions
✅ Generate detailed AI-based feedback reports
✅ Store and manage resumes in MongoDB
✅ Visual analytics dashboard (selected vs rejected)
✅ Email notifications to candidates (selected / improvement suggestions)

---

## 🏗️ Tech Stack

| Component       | Technology                            |
| --------------- | ------------------------------------- |
| Backend         | Flask (Python 3.10+)                  |
| Database        | MongoDB                               |
| AI Model        | OpenAI GPT-3.5 / GPT-4 (configurable) |
| Text Extraction | PyMuPDF / python-docx                 |
| Email Service   | Flask-Mail or SMTP                    |
| Frontend        | HTML + Bootstrap Templates            |


---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate       # (Windows)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```env
FLASK_ENV=development
SECRET_KEY=supersecretkey

UPLOAD_FOLDER=./uploads

MONGO_URI=mongodb://localhost:27017
DB_NAME=resume_db

OPENAI_MODEL=gpt-3.5-turbo

MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_email_password
MAIL_USE_TLS=True
MAIL_DEFAULT_SENDER=your_email@gmail.com
```

> ⚠️ Note: Ensure MongoDB is running locally or remotely.

---

## 🧠 How It Works

1️⃣ **Upload Resume**
The user uploads a `.pdf` or `.docx` file through the web interface.

2️⃣ **Text Extraction**
The system extracts and cleans text using `extract_text()` in `resume_processor.py`.

3️⃣ **AI Analysis**
The extracted text and job description are passed to the **OpenAI model**, which:

* Identifies missing skills
* Calculates a compatibility score (0–10)
* Generates improvement suggestions

4️⃣ **Storage**
All results are stored in MongoDB with timestamps and processed status.

5️⃣ **Email Notification**
After processing:

* If `final_score >= 7.5` → **Selection email**
* Else → **Improvement feedback email**

---

## 📨 Email Templates

### ✅ Selected Candidate Email

**Subject:** Congratulations! You’ve been shortlisted 🎉
**Body:**

```
Dear [Candidate Name],

Congratulations! Your resume has been reviewed and matched successfully with our job requirements.
Our team will contact you shortly for the next steps.

Best regards,
AI Resume Analyzer Team
```

### ⚠️ Improvement Feedback Email

**Subject:** Feedback on your Resume Application 🧩
**Body:**

```
Dear [Candidate Name],

Thank you for applying. Our AI analyzer found areas for improvement in your resume.
We recommend adding more details in the following areas:

- Missing technical skills
- Project details
- Quantitative achievements

Please update and reapply!

Best,
AI Resume Analyzer
```

---

## 📊 Dashboard Analytics

The `/results` page visualizes:

* Count of selected vs rejected candidates
* List of all resumes
* Links to AI-generated detailed reports per resume

---

## 🔍 API Endpoints

| Method | Endpoint         | Description                           |
| ------ | ---------------- | ------------------------------------- |
| `GET`  | `/`              | Upload resume & view random job       |
| `GET`  | `/analyze_page`  | Overview page                         |
| `POST` | `/analyze`       | Analyze all unprocessed resumes       |
| `GET`  | `/results`       | Show results dashboard                |
| `GET`  | `/report/<id>`   | View individual analysis report       |
| `POST` | `/upload_resume` | Upload resume via API (JSON response) |
| `POST` | `/clear_db`      | Delete all resumes                    |
| `GET`  | `/reset_resumes` | Reset DB resumes                      |

---

## 💡 Example AI Report (JSON)

```json
{
  "final_score": 8.2,
  "summary": "Strong alignment with AI Engineer requirements.",
  "feature_scores": {
    "technical_skills": 8.5,
    "experience": 8.0,
    "projects": 7.5,
    "communication": 8.0
  },
  "ai_report": "The candidate has solid Python and ML skills, but should expand on MLOps experience."
}
```

---

## 🧩 Future Enhancements

* ✅ Integration with Gmail API for automatic notifications
* ✅ Candidate dashboard with login
* ✅ Real-time skill gap visualizer
* ✅ Multi-model scoring using Cohere or HuggingFace

---

## 🧑‍💻 Contributing

Pull requests are welcome!
Please fork the repo and submit your changes via PR.

---
