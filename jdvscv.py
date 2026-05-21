import re  
import tkinter as tk  
from tkinter import messagebox  
import spacy  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  

# Load spaCy model  
nlp = spacy.load("en_core_web_sm")  

class CVAnalyzer:  
    def __init__(self, root):  # Fixed constructor  
        self.root = root  
        self.cv_text = ""
        self.job_description_text = ""  

        # Create GUI elements  
        self.label_cv = tk.Label(root, text="Enter CV Text:")  
        self.text_area_cv = tk.Text(root, height=10, width=60)  
        self.label_job = tk.Label(root, text="Enter Job Description:")  
        self.text_area_job = tk.Text(root, height=10, width=60)  
        self.button = tk.Button(root, text="Analyze CV", command=self.analyze_cv)  
        self.result_label = tk.Label(root, text="Result:")  

        # Layout GUI elements  
        self.label_cv.pack()  
        self.text_area_cv.pack()  
        self.label_job.pack()  
        self.text_area_job.pack()  
        self.button.pack()  
        self.result_label.pack()  

    def preprocess_text(self, text):  
        # Tokenization, lemmatization, and removal of stopwords and punctuation  
        doc = nlp(text.lower())  
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]  
        return ' '.join(tokens)  

    def analyze_cv(self):  
        self.cv_text = self.text_area_cv.get("1.0", tk.END).strip()  
        self.job_description_text = self.text_area_job.get("1.0", tk.END).strip()  

        # Preprocess texts  
        preprocessed_cv = self.preprocess_text(self.cv_text)  
        preprocessed_job = self.preprocess_text(self.job_description_text)  

        # Vectorization and similarity check  
        if preprocessed_cv and preprocessed_job:  # Ensure both texts are not empty  
            vectorizer = TfidfVectorizer()  
            vectors = vectorizer.fit_transform([preprocessed_cv, preprocessed_job])  

            # Calculate cosine similarity  
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]  

            # Display similarity score  
            result_text = f"Similarity: {similarity:.2f}"  
        else:  
            result_text = "Please enter both CV and Job Description."  
        
        self.result_label.config(text=result_text)  

if __name__ == "__main__":  # Fixed main execution check  
    root = tk.Tk()  
    app = CVAnalyzer(root)  
    root.mainloop()