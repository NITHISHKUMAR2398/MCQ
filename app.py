import streamlit as st
import time
import random
import numpy as np
from textwrap3 import wrap
import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import traceback
from flashtext import KeywordProcessor
from sense2vec import Sense2Vec
from summarizer import Summarizer
from keyword_extractor import KeywordExtractor
from question_generator import QuestionGenerator
from distractor_generator import DistractorGenerator

class QAGenerationApp:
    def __init__(self):
        self.summarizer = Summarizer()
        self.keyword_extractor = KeywordExtractor()
        self.question_generator = QuestionGenerator()
        self.distractor_generator = DistractorGenerator()

    def run(self):
        placeholder = st.empty()

        with placeholder.container():
            st.title("Question and answer generation:")
            st.subheader("Given an Input of text: ")

            data = st.text_input("Enter your text :  ")
            btn = st.button("Proceed")

        input_txt = data

        if btn:
            time.sleep(2)
            placeholder.empty()

            placeholder2 = st.empty()
            with placeholder2.container():
                summarized_text = self.summarizer.summarizer(input_txt)
                imp_keywords = self.keyword_extractor.get_keywords(input_txt, summarized_text)
                final_qs = []
                final_ans = []
                final_other_opns = []

                for answer in imp_keywords:
                    ques = self.question_generator.get_question(summarized_text, answer)
                    final_qs.append(ques)
                    final_ans.append(answer.capitalize())
                    print("\n")

                for i in range(len(final_qs)):
                    sent = final_qs[i]
                    keyword = final_ans[i]
                    a = self.distractor_generator.get_distractors(keyword, sent, 40, 0.2)

                    if a == []:
                        word = keyword
                        word = word.lower()
                        syns = wn.synsets(word,'n')

                        opns = self.distractor_generator.get_distractors_wordnet(keyword)[:4]
                    else:
                        opns = a
                    final_other_opns.append(opns)

                st.title("Output : ")
                st.subheader("")
                st.subheader("")
                st.subheader("Original : ")
                st.subheader("")
                st.subheader("")
                st.text(input_txt)
                st.subheader("")
                st.subheader("")
                st.subheader("Summarised : ")
                st.subheader("")
                st.subheader("")
                st.text(summarized_text)
                st.subheader("")
                st.subheader("")
                st.subheader("Important Keywords : ")
                st.subheader("")
                st.subheader("")
                st.text(imp_keywords)
                st.subheader("")
                st.subheader("")
                st.subheader("MCQ : ")
                st.subheader("")
                st.subheader("")
                for itrt in range(len(final_qs)):
                    tmp1 = final_other_opns[itrt]
                    tmp2 = final_ans[itrt]
                    fin_ans_op = tmp1 + [tmp2]
                    random.shuffle(fin_ans_op)
                    option = st.selectbox(final_qs[itrt], fin_ans_op)

                st.title("Answers")
                for i in range(len(final_qs)):
                    st.text(final_qs[i])
                    st.text(final_ans[i])

if __name__ == "__main__":
    app = QAGenerationApp()
    app.run()