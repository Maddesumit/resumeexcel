from flask import Flask, request, render_template
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


# Function to extract and normalize skills from the input text
def extract_skills(text):
    """Extracts and normalizes skills from the resume text."""
    if not text:
        return []

    # Normalize the text to lowercase and remove any non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Remove special characters and digits
    return text.split()


@app.route("/")
def home():
    return render_template('matchresume.html')


@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        # Get HR input from form
        required_skills = request.form['required_skills'].split(",")  # Skills entered by the HR
        required_experience = int(request.form['required_experience'])
        required_education = request.form['required_education']
        file = request.files['excel_file']

        if not file:
            return render_template('matchresume.html', message="Please upload a valid Excel file.")

        # Read Excel file
        try:
            df = pd.read_excel(file)
            if not {'Name', 'Skills', 'Experience', 'Education'}.issubset(df.columns):
                return render_template('matchresume.html',
                                       message="Excel file must contain 'Name', 'Skills', 'Experience', and 'Education' columns.")
        except Exception as e:
            return render_template('matchresume.html', message=f"Error reading Excel file: {e}")

        # Combine 'Skills' and 'Experience' for vectorization
        resumes = df['Skills'] + " " + df['Experience']
        vectorizer = TfidfVectorizer().fit_transform([" ".join(required_skills)] + resumes.tolist())
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # Generate feedback for each candidate
        feedback = []
        for i, similarity in enumerate(similarities):
            # Extract skills from resume
            resume_skills = extract_skills(df.iloc[i]['Skills'])  # Extract skills from resume

            # Calculate number of matched skills
            matched_skills_count = len([skill for skill in required_skills if skill.strip().lower() in resume_skills])
            total_skills_count = len(required_skills)

            # Check experience and education
            try:
                resume_experience = int(df.iloc[i]['Experience'].split()[0])  # Extract years of experience
            except ValueError:
                resume_experience = 0  # Handle cases where experience isn't clear
            experience_feedback = (
                f"Meets experience requirement ({resume_experience} years)"
                if resume_experience >= required_experience else f"Does not meet experience requirement ({resume_experience} years)"
            )
            education_feedback = (
                f"Meets education requirement ({df.iloc[i]['Education']})"
                if required_education.lower() in df.iloc[i][
                    'Education'].lower() else f"Does not meet education requirement ({df.iloc[i]['Education']})"
            )

            # Generate overall feedback
            feedback.append({
                'name': df.iloc[i]['Name'],
                'similarity_score': round(similarity, 2),
                'skills_feedback': f"Matched {matched_skills_count} out of {total_skills_count} skills.",
                'experience_feedback': experience_feedback,
                'education_feedback': education_feedback,
            })

        # Sort feedback by similarity score in descending order
        feedback = sorted(feedback, key=lambda x: x['similarity_score'], reverse=True)

        return render_template(
            'matchresume.html',
            message="Top matching resumes:",
            feedback=feedback
        )

    return render_template('matchresume.html')


if __name__ == '__main__':
    app.run(debug=True)
