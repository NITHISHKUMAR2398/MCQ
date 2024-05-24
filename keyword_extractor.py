import string
import pke
import traceback
from flashtext import KeywordProcessor
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

class KeywordExtractor:
    def __init__(self):
        pass

    def get_nouns_multipartite(self, content):
        out = []
        try:
            extractor = pke.unsupervised.MultipartiteRank()
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            extractor.load_document(input=content, stoplist=stoplist)
            pos = {'PROPN', 'NOUN', 'ADJ'}
            extractor.candidate_selection(pos=pos)

            extractor.candidate_weighting(alpha=1.1,
                                          threshold=0.74,
                                          method='average')

            keyphrases = extractor.get_n_best(n=12)

            for val in keyphrases:
                out.append(val[0])
        except:
            out = []
            traceback.print_exc()
        return out

    def get_keywords(self, input_txt, summarized_text):
        keywords = self.get_nouns_multipartite(input_txt)
        print("keywords unsummarized: ", keywords)
        keyword_processor = KeywordProcessor()
        for keyword in keywords:
            keyword_processor.add_keyword(keyword)

        keywords_found = keyword_processor.extract_keywords(summarized_text)
        keywords_found = list(set(keywords_found))
        print("keywords_ found in summarized: ", keywords_found)

        important_keywords = []
        for keyword in keywords:
            if keyword in keywords_found:
                important_keywords.append(keyword)
        return important_keywords