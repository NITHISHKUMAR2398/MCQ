import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import sent_tokenize
class Summarizer:
    def __init__(self):
        self.summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_model = self.summary_model.to(self.device)

    def post_process_text(self, content):
        final = ""
        for sent in sent_tokenize(content):
            sent = sent.capitalize()
            final = final + " " + sent
        return final

    def summarizer(self, text):
        text = text.strip().replace("\n", " ")
        text = "summarize: " + text
        max_len = 512
        encoding = self.summary_tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt").to(self.device)

        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outs = self.summary_model.generate(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           early_stopping=True,
                                           num_beams=3,
                                           num_return_sequences=1,
                                           no_repeat_ngram_size=2,
                                           min_length=75,
                                           max_length=300)

        dec = [self.summary_tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        summary = dec[0]
        summary = self.post_process_text(summary)
        summary = summary.strip()
        return summary