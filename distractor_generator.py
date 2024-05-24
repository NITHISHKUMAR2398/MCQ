from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from similarity.normalized_levenshtein import NormalizedLevenshtein
import numpy as np
from nltk.corpus import wordnet as wn

class DistractorGenerator:
    def __init__(self):
        self.s2v = Sense2Vec().from_disk("s2v_old")
        self.sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')
        self.normalized_levenshtein = NormalizedLevenshtein()

    def filter_same_sense_words(self, original, wordlist):
        filtered_words = []
        base_sense = original.split('|')[1]
        print(base_sense)
        for eachword in wordlist:
            if eachword[0].split('|')[1] == base_sense:
                filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
        return filtered_words

    def get_highest_similarity_score(self, wordlist, wrd):
        score = []
        for each in wordlist:
            score.append(self.normalized_levenshtein.similarity(each.lower(), wrd.lower()))
        return max(score)

    def sense2vec_get_words(self, word, topn, question):
        output = []
        print("word ", word)
        try:
            sense = self.s2v.get_best_sense(word, senses=["NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT", "NORP", "WORK OF ART", "FAC", "GPE", "NUM", "FACILITY"])
            most_similar = self.s2v.most_similar(sense, n=topn)
            output = self.filter_same_sense_words(sense, most_similar)
            print("Similar ", output)
        except:
            output = []

        threshold = 0.6
        final = [word]
        checklist = question.split()
        for x in output:
            if self.get_highest_similarity_score(final, x) < threshold and x not in final and x not in checklist:
                final.append(x)

        return final[1:]

    def mmr(self, doc_embedding, word_embeddings, words, top_n, lambda_param):
        word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
        word_similarity = cosine_similarity(word_embeddings)
        keywords_idx = [np.argmax(word_doc_similarity)]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        for _ in range(top_n - 1):
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
            mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        return [words[idx] for idx in keywords_idx]

    def get_distractors_wordnet(self, word):
        distractors = []
        try:
            syn = wn.synsets(word, 'n')[0]

            word = word.lower()
            orig_word = word
            if len(word.split()) > 0:
                word = word.replace(" ", "_")
            hypernym = syn.hypernyms()
            if len(hypernym) == 0:
                return distractors
            for item in hypernym[0].hyponyms():
                name = item.lemmas()[0].name()
                if name == orig_word:
                    continue
                name = name.replace("_", " ")
                name = " ".join(w.capitalize() for w in name.split())
                if name is not None and name not in distractors:
                    distractors.append(name)
        except:
            print("Wordnet distractors not found")
        return distractors

    def get_distractors(self, word, orig_sentence, top_n, lambda_val):
        distractors = self.sense2vec_get_words(word, top_n, orig_sentence)
        print("distractors ", distractors)
        if len(distractors) == 0:
            return distractors
        distractors_new = [word.capitalize()]
        distractors_new.extend(distractors)

        embedding_sentence = orig_sentence + " " + word.capitalize()
        keyword_embedding = self.sentence_transformer_model.encode([embedding_sentence])
        distractor_embeddings = self.sentence_transformer_model.encode(distractors_new)

        max_keywords = min(len(distractors_new), 5)
        filtered_keywords = self.mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambda_val)

        final = [word.capitalize()]
        for wrd in filtered_keywords:
            if wrd.lower() != word.lower():
                final.append(wrd.capitalize())
        final = final[1:]
        return final