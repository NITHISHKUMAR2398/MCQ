Question and Answer Generation App

This application generates questions and answers based on an input text using natural language processing techniques. It summarizes the input text, extracts important keywords, generates questions based on the keywords, and provides distractors for each question.

Create an new virtual environment
python -m venv venv
source venv/scripts/activate
Install the required dependencies by running:
pip install -r requirements.txt
To use the application:
a. Run the main script app.py:

streamlit run app.py
b. Enter the input text in the provided text input box and click on the "Proceed" button.

c. The application will then summarize the input text, extract important keywords, generate questions based on the keywords, and provide distractors for each question.

d. You can view the original text, summarized text, important keywords, generated questions, and answers in the application interface.

app.py: Contains the main application logic and Streamlit UI.

summarizer.py: Contains the Summarizer class responsible for text summarization.

keyword_extractor.py: Contains the KeywordExtractor class responsible for keyword extraction.

question_generator.py: Contains the QuestionGenerator class responsible for question generation.

distractor_generator.py: Contains the DistractorGenerator class responsible for distractor generation.

Note: Use Python 3.10.0 to satisfy all dependancies. Ensure you have the pre-trained models for t5-base and ramsrigouthamg/t5_squad_v1. If not, they will be automatically downloaded when the application is run for the first time.

All the required files are attached in Google Drive: https://drive.google.com/drive/folders/1QNliz5lnj8s86-lk2L8nc_Tr5KtpJMU0?usp=sharing