# ⚡ AI Talent Intelligence (Resume Screener)

An advanced, modern web application designed to automate and optimize the talent screening process. This tool uses Natural Language Processing (NLP) and Machine Learning to parse resumes (PDF and DOCX), extract key candidate entities, and calculate a suitability score based on predefined job roles and required skills.

## 🚀 Key Features

* **Automated Parsing:** Seamlessly extracts text from multiple document formats (`.pdf` and `.docx`).
* **Intelligent Entity Extraction:** Uses `spaCy` to automatically detect candidate names, email addresses, and phone numbers.
* **Smart Matching Algorithm:** Leverages `scikit-learn` (TF-IDF Vectorizer and Cosine Similarity) to calculate an accurate "Match %" between the candidate's resume and the target job profile.
* **Pre-configured Role Database:** Includes a comprehensive, searchable database of standard tech and business roles with associated industry skills.
* **Interactive Analytics:** Generates real-time, interactive data visualizations (bar charts and pie charts) using `Plotly` to map out candidate suitability and skill coverage.
* **Adaptive Glassmorphism UI:** Features a highly modern, responsive layout that automatically adapts to the user's system Light/Dark mode preferences.

## 🛠️ Tech Stack

* **Language:** Python 3.12+
* **Frontend/Framework:** Streamlit
* **Natural Language Processing:** spaCy (`en_core_web_sm`)
* **Machine Learning:** Scikit-Learn
* **Data Manipulation:** Pandas
* **Data Visualization:** Plotly Express
* **Document Parsing:** PyPDF, python-docx

## 💻 Local Installation

To run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/ai-resume-screener.git](https://github.com/your-username/ai-resume-screener.git)
   cd ai-resume-screener