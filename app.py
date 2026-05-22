import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

import io
import numpy as np
from datetime import datetime
import unicodedata

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

st.set_page_config(page_title="AI Resume Analyzer", page_icon="assets/document_icon.png", layout="wide")
st.title("AI Resume Analyzer Pro")
st.markdown("""
Upload your resume (PDF) and paste a job description.  
This tool uses **Sentence-BERT semantic similarity**, **NER-based skill extraction**, 
**zero-shot domain classification**, **skill co-occurrence graph**, and **employment gap detection**.
""")

with st.sidebar:
    st.header("About")
    st.info("""
**What makes this unique:**
- SBERT semantic matching — understands *meaning*, not just keywords
- spaCy NER — extracts skills from context intelligently  
- Skill co-occurrence graph — recommends what to learn next
- Employment gap detector — parses dates to find timeline gaps
- Zero-shot domain classifier — no labeled training data needed
""")
    st.header("How It Works")
    st.write("""
1. Upload resume (PDF).
2. Paste the job description.
3. Click **Analyze**.
4. Get deep ML-powered insights.
""")

# ─── SKILL DATA ───────────────────────────────────────────────────────────────
SKILL_KEYWORDS = {
    "python","java","c++","c","sql","excel","tableau","powerbi","r",
    "pandas","numpy","scikit-learn","machine learning","deep learning",
    "tensorflow","pytorch","nlp","keras","spark","hadoop",
    "html","css","javascript","react","angular","vue","bootstrap",
    "django","flask","fastapi","aws","azure","gcp","docker",
    "kubernetes","git","github","linux","bash","nosql","mongodb",
    "postgresql","mysql","analytics","visualization","etl",
    "testing","llm","transformers","bert","gpt","langchain",
    "reinforcement learning","computer vision","opencv","mlops",
}

CO_OCCURRENCE = {
    "python": ["pandas","numpy","scikit-learn","flask","django","fastapi"],
    "machine learning": ["python","tensorflow","pytorch","scikit-learn","nlp","deep learning"],
    "deep learning": ["tensorflow","pytorch","keras","nlp","computer vision"],
    "sql": ["postgresql","mysql","mongodb","nosql","analytics"],
    "aws": ["docker","kubernetes","linux","gcp","azure"],
    "react": ["javascript","html","css","angular","vue"],
    "docker": ["kubernetes","aws","linux","bash","gcp"],
    "nlp": ["python","transformers","bert","llm","langchain"],
}

DOMAIN_PROFILES = {
    "Data Science / ML": {"python","machine learning","deep learning","nlp","pandas","tensorflow","pytorch","scikit-learn"},
    "Web Development":   {"html","css","javascript","react","django","flask","angular","vue"},
    "Cloud / DevOps":    {"aws","azure","gcp","docker","kubernetes","linux","bash"},
    "Data Analytics":    {"sql","excel","tableau","powerbi","analytics","etl","visualization"},
    "AI / LLM Eng":      {"llm","transformers","bert","langchain","gpt","nlp"},
}

QUESTION_TEMPLATES = {
    "python":           ["Explain Python decorators.", "How do you manage memory in Python?"],
    "sql":              ["INNER JOIN vs LEFT JOIN?", "How would you optimize a slow query?"],
    "machine learning": ["Bias-variance tradeoff?", "How do you handle class imbalance?"],
    "deep learning":    ["Explain backpropagation.", "When to use LSTM vs Transformer?"],
    "docker":           ["Docker image vs container?"],
    "javascript":       ["Explain event delegation."],
}

MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
}

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def extract_text_from_pdf(f):
    try:
        r = PyPDF2.PdfReader(f)
        return "".join(p.extract_text() or "" for p in r.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-zA-Z\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()

def remove_stopwords(t):
    sw = set(stopwords.words("english"))
    return " ".join(w for w in word_tokenize(t) if w not in sw)

def tfidf_sim(rt, jt):
    r = remove_stopwords(clean_text(rt))
    j = remove_stopwords(clean_text(jt))
    v = TfidfVectorizer()
    m = v.fit_transform([r, j])
    return round(cosine_similarity(m[0:1], m[1:2])[0][0]*100, 2), r, j

def normalize_for_pdf(text):
    if text is None:
        return ""
    t = str(text)
    t = unicodedata.normalize("NFKD", t)
    t = t.replace("—", "-").replace("–", "-").replace("…", "...")
    return t.encode("latin-1", errors="replace").decode("latin-1")

def sbert_sim(rt, jt):
    if not HAS_SBERT:
        return None
    r_e = _sbert_model.encode([rt])
    j_e = _sbert_model.encode([jt])
    return round(float(cosine_similarity(r_e, j_e)[0][0])*100, 2)

def blended(tf, sb):
    return round(0.35*tf + 0.65*sb, 2) if sb else tf

def extract_skills(text):
    if HAS_SPACY:
        doc = _nlp(text)
        found = set()
        for chunk in doc.noun_chunks:
            t = chunk.text.lower().strip()
            if t in SKILL_KEYWORDS:
                found.add(t)
            for kw in SKILL_KEYWORDS:
                if kw in t and len(kw) > 4:
                    found.add(kw)
        for token in doc:
            if token.text.lower() in SKILL_KEYWORDS:
                found.add(token.text.lower())
        return sorted(found)
    else:
        tokens = re.findall(r"[A-Za-z+#\-]{2,}", text.lower())
        return sorted({t for t in tokens if t in SKILL_KEYWORDS})

def recommend_next(skills, n=5):
    cur = set(skills)
    counts = defaultdict(int)
    for s in skills:
        for nb in CO_OCCURRENCE.get(s, []):
            if nb not in cur:
                counts[nb] += 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]

def draw_graph(cur_skills, rec):
    G = nx.Graph()
    rec_s = [s for s,_ in rec]
    all_n = set(cur_skills) | set(rec_s)
    for s in cur_skills:
        for nb in CO_OCCURRENCE.get(s, []):
            if nb in all_n:
                G.add_edge(s, nb)
    if len(G.nodes) == 0:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    colors = ["#0f9d58" if n in cur_skills else "#ffa726" for n in G.nodes]
    nx.draw_networkx(G, pos, ax=ax, node_color=colors, node_size=1200,
                     font_size=8, font_weight="bold", edge_color="#ccc", width=1.5)
    ax.legend(handles=[
        mpatches.Patch(color="#0f9d58", label="Your skills"),
        mpatches.Patch(color="#ffa726", label="Recommended next"),
    ], loc="lower left", fontsize=9)
    ax.set_title("Skill Co-occurrence Graph", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig

def parse_month_year(s):
    m = re.match(r"([a-z]+)\.?\s+(\d{4})", s.strip(), re.I)
    if not m:
        return None
    mon = MONTH_MAP.get(m.group(1).lower()[:3])
    return (int(m.group(2)), mon) if mon else None

def extract_date_ranges(text):
    pat = re.compile(
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{4})"
        r"\s*[–\-—to]+\s*"
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{4}|present|current)",
        re.IGNORECASE)
    ranges = []
    for m in pat.finditer(text):
        start = parse_month_year(m.group(1))
        end_s = m.group(2).strip()
        end = (datetime.now().year, datetime.now().month) if end_s.lower() in ("present","current") else parse_month_year(end_s)
        if start and end:
            ranges.append((start, end))
    return sorted(ranges)

def find_gaps(ranges, threshold=3):
    gaps = []
    for i in range(len(ranges)-1):
        _, e = ranges[i]
        s, _ = ranges[i+1]
        gap = (s[0]*12+s[1]) - (e[0]*12+e[1])
        if gap > threshold:
            gaps.append({"after": f"{e[1]:02d}/{e[0]}", "before": f"{s[1]:02d}/{s[0]}", "months": gap})
    return gaps

def classify_domain(skills):
    ss = set(skills)
    scores = {d: round(len(ss & p)/max(len(ss|p),1)*100,1) for d,p in DOMAIN_PROFILES.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def check_sections(text):
    secs = {
        "Skills":         bool(re.search(r"\bskills\b", text, re.I)),
        "Projects":       bool(re.search(r"\bprojects\b", text, re.I)),
        "Education":      bool(re.search(r"\beducation\b", text, re.I)),
        "Experience":     bool(re.search(r"\bexperience\b", text, re.I)),
        "Certifications": bool(re.search(r"\b(certif|certifications|certificate)\b", text, re.I)),
    }
    missing = [k for k,v in secs.items() if not v]
    return secs, missing, int((1 - len(missing)/len(secs))*100)

def smart_feedback(text, skills, score, missing_secs):
    tips = []
    if missing_secs:
        tips.append("Add missing sections: " + ", ".join(missing_secs))
    if not re.search(r"\b\d+%|\b\d+\s+(?:months|years|people|users|clients)\b", text):
        tips.append("Add measurable achievements (numbers, percentages, impact).")
    if not re.search(r"github\.com|bitbucket\.org", text, re.I):
        tips.append("Include GitHub or project links for technical work.")
    if score < 50:
        tips.append("Tailor your resume keywords to match the JD more closely.")
    if len(skills) < 3:
        tips.append("List more skills and technologies you are comfortable with.")
    if not tips:
        tips.append("Resume looks great — focus on clarity and quantifying impact.")
    return tips

def ats_score(sim, rp, jp, comp, skills, miss):
    jw, rw = set(jp.split()), set(rp.split())
    kw = len(jw & rw) / max(len(jw), 1)
    sm = min(len(skills)/10, 1.0)
    mp = max(0, len(miss)/max(len(jw), 1))
    s = 0.4*(sim/100) + 0.3*kw + 0.15*(comp/100) + 0.15*sm
    return int(round(s*(1-0.5*mp)*100))

def gen_questions(skills, jd, max_q=10):
    qs = []
    for s in skills:
        if s in QUESTION_TEMPLATES:
            qs.extend(QUESTION_TEMPLATES[s])
        else:
            qs.append(f"Explain {s} and how you have used it in a project.")
        if len(qs) >= max_q:
            break
    jd_toks = set(re.findall(r"[A-Za-z]{3,}", jd.lower()))
    if "lead" in jd_toks or "manager" in jd_toks:
        qs.append("Describe a time you led a technical project.")
    if "team" in jd_toks:
        qs.append("How do you collaborate with cross-functional teams?")
    return qs[:max_q]

def make_pdf(data):
    if not HAS_FPDF:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, normalize_for_pdf("AI Resume Analyzer Pro - Report"), ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, normalize_for_pdf(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"), ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, normalize_for_pdf(f"Blended Match Score: {data['blended']}%"), ln=True)
    pdf.cell(0, 8, normalize_for_pdf(f"ATS Score: {data['ats']}/100"), ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", "B", 12); pdf.cell(0, 8, normalize_for_pdf("Domain Classification:"), ln=True)
    pdf.set_font("Arial", "", 11)
    for d,s in data.get("domains",[])[:3]:
        pdf.cell(0, 6, normalize_for_pdf(f"  {d}: {s}%"), ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12); pdf.cell(0, 8, normalize_for_pdf("Recommended Skills to Learn:"), ln=True)
    pdf.set_font("Arial", "", 11)
    for sk,c in data.get("next_skills",[]):
        pdf.cell(0, 6, normalize_for_pdf(f"  {sk} (linked to {c} of your skills)"), ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12); pdf.cell(0, 8, normalize_for_pdf("Employment Gaps:"), ln=True)
    pdf.set_font("Arial", "", 11)
    for g in data.get("gaps", []):
        pdf.cell(0, 6, normalize_for_pdf(f"  {g['months']}mo gap: {g['after']} to {g['before']}"), ln=True)
    if not data.get("gaps"):
        pdf.cell(0, 6, normalize_for_pdf("  No significant gaps detected."), ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12); pdf.cell(0, 8, normalize_for_pdf("Feedback:"), ln=True)
    pdf.set_font("Arial", "", 11)
    for f in data.get("feedback",[]):
        pdf.multi_cell(0, 6, normalize_for_pdf(f"  - {f}"))
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12); pdf.cell(0, 8, normalize_for_pdf("Interview Questions:"), ln=True)
    pdf.set_font("Arial", "", 11)
    for q in data.get("questions",[]):
        pdf.multi_cell(0, 6, normalize_for_pdf(f"  - {q}"))
    buf = io.BytesIO()
    buf.write(pdf.output(dest="S").encode("latin-1", errors="replace"))
    buf.seek(0)
    return buf.read()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    job_description = st.text_area("Paste the job description", height=200)

    c1, c2, c3 = st.columns(3)
    c1.metric("SBERT Engine", "Active" if HAS_SBERT else "Fallback to TF-IDF")
    c2.metric("spaCy NER", "Active" if HAS_SPACY else "Fallback to regex")
    c3.metric("Graph Engine", "NetworkX Active")

    if not st.button("Analyze Resume"):
        return
    if not uploaded_file:
        st.warning("Please upload your resume.")
        return
    if not job_description.strip():
        st.warning("Please paste the job description.")
        return

    with st.spinner("Running ML analysis pipeline..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        if not resume_text:
            st.error("Could not extract text from PDF.")
            return

        # 1. Similarity
        tf_score, res_proc, jd_proc = tfidf_sim(resume_text, job_description)
        sb_score = sbert_sim(resume_text, job_description)
        blend = blended(tf_score, sb_score)

        st.subheader("Similarity Analysis")
        a, b_, c_ = st.columns(3)
        a.metric("TF-IDF (surface)", f"{tf_score}%")
        if sb_score:
            b_.metric("SBERT (semantic)", f"{sb_score}%")
        c_.metric("Blended Score", f"{blend}%")

        fig_g, ax_g = plt.subplots(figsize=(7, 0.6))
        color = "#ff4b4b" if blend < 40 else "#ffa726" if blend < 70 else "#0f9d58"
        ax_g.barh([0], blend, color=color, height=0.5)
        ax_g.set_xlim(0, 100); ax_g.set_xlabel("Score (%)"); ax_g.set_yticks([])
        ax_g.set_title("Blended Resume-JD Match Score")
        st.pyplot(fig_g)

        if blend < 40:
            st.warning("Low match — tailor your resume more closely to the JD.")
        elif blend < 70:
            st.info("Moderate match — targeted improvements recommended.")
        else:
            st.success("Strong semantic match!")

        # 2. NER Skill Extraction
        res_skills = extract_skills(resume_text)
        jd_skills  = extract_skills(job_description)
        all_skills = sorted(set(res_skills) | set(jd_skills))

        st.subheader("Skills Detected (NER)")
        sc1, sc2 = st.columns(2)
        sc1.markdown("**In your resume:**\n" + (", ".join(res_skills) or "None detected"))
        sc2.markdown("**Required by JD:**\n" + (", ".join(jd_skills) or "None detected"))
        miss_jd = sorted(set(jd_skills) - set(res_skills))
        if miss_jd:
            st.error("Skills in JD but missing from resume: " + ", ".join(miss_jd))
        else:
            st.success("All detected JD skills are present in your resume!")

        # 3. Zero-shot Domain Classification
        st.subheader("Zero-Shot Domain Classification")
        domains = classify_domain(all_skills)
        names = [d for d,_ in domains[:5]]
        dscores = [s for _,s in domains[:5]]
        df_fig, df_ax = plt.subplots(figsize=(7, 3))
        bars = df_ax.barh(names[::-1], dscores[::-1], color="#4285F4")
        df_ax.set_xlabel("Domain Match (%)"); df_ax.set_title("Best-Fit Job Domains")
        df_ax.bar_label(bars, fmt="%.0f%%", padding=3)
        plt.tight_layout(); st.pyplot(df_fig)

        # 4. Skill Co-occurrence Graph
        st.subheader("Skill Co-occurrence Graph & Learning Path")
        next_skills = recommend_next(res_skills)
        g_fig = draw_graph(res_skills, next_skills)
        if g_fig:
            st.pyplot(g_fig)
        if next_skills:
            st.markdown("**Recommended next skills to learn:**")
            for sk, cnt in next_skills:
                st.write(f"- **{sk}** — connected to {cnt} of your existing skills")

        # 5. Employment Gap Detector
        st.subheader("Employment Gap Detector")
        date_ranges = extract_date_ranges(resume_text)
        gaps = find_gaps(date_ranges)
        if date_ranges:
            st.write(f"Found **{len(date_ranges)}** date range(s) in resume.")
            if gaps:
                for g in gaps:
                    st.warning(f"{g['months']}-month gap between {g['after']} and {g['before']} — prepare an explanation.")
            else:
                st.success("No significant employment gaps detected.")
        else:
            st.info("No date ranges found — use formats like 'Jan 2022 – Mar 2023'.")

        # 6. Resume Completeness
        secs_p, miss_secs, comp = check_sections(resume_text)
        st.subheader("Resume Completeness")
        cc1, cc2 = st.columns(2)
        with cc1:
            for sec, present in secs_p.items():
                st.write(f"{sec}: {'✅' if present else '❌'}")
        with cc2:
            st.progress(comp); st.caption(f"Completeness: {comp}%")

        # 7. ATS Score
        miss_kw = list(set(jd_proc.split()) - set(res_proc.split()))[:15]
        ats = ats_score(blend, res_proc, jd_proc, comp, res_skills, miss_kw)
        st.subheader("ATS Score")
        st.metric("ATS Score", f"{ats}/100")

        # 8. Feedback
        st.subheader("Smart Resume Feedback")
        feedback = smart_feedback(resume_text, res_skills, blend, miss_secs)
        for f in feedback:
            st.info(f)

        # 9. Interview Questions
        st.subheader("Likely Interview Questions")
        questions = gen_questions(all_skills, job_description)
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")

        # 10. PDF Report
        st.subheader("Download Full Report")
        pdf_bytes = make_pdf({
            "blended": blend, "ats": ats, "domains": domains,
            "next_skills": next_skills, "gaps": gaps,
            "feedback": feedback, "questions": questions,
        })
        if pdf_bytes:
            st.download_button("Download PDF Report", data=pdf_bytes,
                               file_name="resume_analysis_report.pdf", mime="application/pdf")
        else:
            st.caption("Install `fpdf` to enable PDF download.")

if __name__ == "__main__":
    main()
