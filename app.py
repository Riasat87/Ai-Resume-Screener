import re
import spacy
import docx
import pandas as pd
import streamlit as st
import plotly.express as px
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screener", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

adaptive_css = """
<style>
    .gradient-text {
        background: linear-gradient(to right, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        background-size: 200% 200%;
        text-align: center;
        font-weight: 900;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: #8b5cf6;
        box-shadow: 0 20px 25px -5px rgba(139, 92, 246, 0.2);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 900;
        color: var(--primary-color);
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 10px 25px -5px rgba(139, 92, 246, 0.5);
        transform: translateY(-2px);
    }
    .empty-state {
        border: 2px dashed rgba(128, 128, 128, 0.3);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        background: rgba(128, 128, 128, 0.05);
        margin-top: 2rem;
    }
</style>
"""
st.markdown(adaptive_css, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_model()

JOB_ROLES = sorted([
    "Software Engineer", "Data Scientist", "Product Manager", "UI/UX Designer", 
    "DevOps Engineer", "QA Automation Engineer", "Business Analyst", 
    "Machine Learning Engineer", "Full Stack Developer", "Cloud Architect", 
    "Cybersecurity Analyst", "Mobile App Developer", "Data Analyst", 
    "Systems Administrator", "Network Engineer", "Database Administrator", 
    "IT Support Specialist", "Scrum Master", "Technical Writer", 
    "Marketing Manager", "Sales Executive", "HR Generalist", "Financial Analyst",
    "Frontend Developer", "Backend Developer", "Data Engineer", "AI Researcher",
    "Game Developer", "Blockchain Developer", "Software Quality Assurance"
])

SKILLS_LIST = sorted([
    "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "Go", "Swift", "Kotlin", "TypeScript", "PHP", "R", "MATLAB", "Rust", "Dart",
    "React", "Angular", "Vue.js", "Node.js", "Express", "Django", "Flask", "Spring Boot", "Ruby on Rails", "HTML", "CSS", "Sass", "Tailwind CSS", "React Native", "Flutter",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "Oracle", "DynamoDB", "Elasticsearch", "SQLite",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "Git", "GitHub", "GitLab", "Terraform", "Ansible", "Linux", "Unix", "CI/CD",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Pandas", "NumPy", "Scikit-Learn", "TensorFlow", "PyTorch", "Hadoop", "Spark", "Kafka", "Tableau", "Power BI", "Data Visualization",
    "Selenium", "Cypress", "JUnit", "PyTest", "Jest", "Postman", "Appium", "Manual Testing", "Test Automation",
    "Figma", "Adobe XD", "Sketch", "Photoshop", "Illustrator", "Agile", "Scrum", "Jira", "Confluence", "Trello", "Wireframing", "Prototyping",
    "Communication", "Leadership", "Problem Solving", "Project Management", "Data Analysis", "SEO", "Content Strategy", "Digital Marketing", "Salesforce", "CRM", "B2B Sales"
])

def extract_text_from_file(uploaded_file):
    try:
        text = ""
        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + " "
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            text = " ".join([paragraph.text for paragraph in doc.paragraphs])
        return re.sub(r'\s+', ' ', text).strip()
    except Exception:
        return ""

def extract_entities(text):
    doc = nlp(text)
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Not Found")
    emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    phones = re.findall(r'\(?\b[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', text)
    return name, emails[0] if emails else "Not Found", phones[0] if phones else "Not Found"

def extract_skills(text, selected_skills):
    text_lower = text.lower()
    return [skill for skill in selected_skills if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower)]

def calculate_similarity(resume_text, target_text):
    if not target_text.strip(): return 0.0
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        vectors = vectorizer.fit_transform([target_text, resume_text])
        return round(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100, 2)
    except ValueError:
        return 0.0

st.markdown("<div class='gradient-text'>AI Talent Intelligence</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.7; font-size: 1.1rem; margin-bottom: 2rem;'>Next-Generation Resume Parsing & Ranking</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎯 Role Requirements")
    
    selected_role = st.selectbox(
        "Job Role", 
        options=JOB_ROLES, 
        help="Select the target position to optimize the AI scoring."
    )
    
    selected_skills = st.multiselect(
        "Required Skills", 
        options=SKILLS_LIST, 
        help="Search and select all required skills."
    )
    
    st.markdown("---")
    st.caption("✨ Fully adaptive to your system's Light/Dark mode.")

st.markdown("### 📄 Drop Resumes Below")
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed")

if st.button("🚀 Analyze Candidates", use_container_width=True):
    if not uploaded_files:
        st.error("Upload at least one resume to run the analysis.")
    elif not selected_role or not selected_skills:
        st.error("Please select a Job Role and at least one Required Skill in the sidebar.")
    else:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        target_job_description = selected_role + " " + " ".join(selected_skills)
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.markdown(f"**🧠 Processing:** `{uploaded_file.name}` ...")
            resume_text = extract_text_from_file(uploaded_file)
            
            if resume_text:
                name, email, phone = extract_entities(resume_text)
                matched_skills = extract_skills(resume_text, selected_skills)
                score = calculate_similarity(resume_text, target_job_description)
                
                results.append({
                    "Match %": score,
                    "Candidate Name": name,
                    "Skills Hit": len(matched_skills),
                    "Matched Skills": ", ".join(matched_skills),
                    "Email": email,
                    "File Type": uploaded_file.name.split('.')[-1].upper()
                })
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.empty()
        progress_bar.empty()
        
        if results:
            st.toast("Analysis Complete!", icon="✅")
            st.balloons()
            
            df = pd.DataFrame(results).sort_values(by="Match %", ascending=False).reset_index(drop=True)
            best_candidate = df.iloc[0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["📈 Visual Analytics", "📋 Interactive Leaderboard"])
            
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🌟 Top Candidate", best_candidate['Candidate Name'], f"{best_candidate['Match %']}% Match")
                col2.metric("📄 Total Processed", len(results), "Resumes")
                col3.metric("🎯 Avg Score", f"{round(df['Match %'].mean(), 1)}%", "Overall")
                col4.metric("💡 Highest Skill Hit", int(df['Skills Hit'].max()), "Skills Found")
                
                st.markdown("<br>", unsafe_allow_html=True)
                chart_col1, chart_col2 = st.columns([2, 1])
                
                with chart_col1:
                    fig_bar = px.bar(
                        df, x="Candidate Name", y="Match %", 
                        color="Match %", color_continuous_scale="Purp",
                        text_auto='.2s', title="Candidate Suitability Ranking"
                    )
                    fig_bar.update_layout(margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with chart_col2:
                    fig_pie = px.pie(
                        df, values="Skills Hit", names="Candidate Name", hole=0.5,
                        color_discrete_sequence=px.colors.sequential.Purp_r,
                        title="Skill Coverage Distribution"
                    )
                    fig_pie.update_layout(margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pie, use_container_width=True)

            with tab2:
                st.dataframe(
                    df,
                    column_config={
                        "Match %": st.column_config.ProgressColumn(
                            "Suitability Score",
                            help="Cosine similarity against role and skills",
                            format="%f%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "Email": st.column_config.TextColumn("Contact Info")
                    },
                    use_container_width=True, 
                    hide_index=True
                )
                
                st.download_button(
                    label="📥 Export Report (CSV)",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='talent_screening_report.csv',
                    mime='text/csv',
                    use_container_width=True
                )
elif not uploaded_files:
    st.markdown("""
    <div class='empty-state'>
        <h2 style='opacity: 0.7;'>Awaiting Candidates...</h2>
        <p style='opacity: 0.5;'>Drag and drop PDF or Word documents into the uploader above to unleash the AI.</p>
    </div>
    """, unsafe_allow_html=True)