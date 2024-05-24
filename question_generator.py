import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class QuestionGenerator:
    def __init__(self):
        self.question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
        self.question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.question_model = self.question_model.to(self.device)

    def get_question(self, context, answer):
        text = "context: {} answer: {}".format(context, answer)
        encoding = self.question_tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(self.device)
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outs = self.question_model.generate(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            early_stopping=True,
                                            num_beams=5,
                                            num_return_sequences=1,
                                            no_repeat_ngram_size=2,
                                            max_length=72)

        dec = [self.question_tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        Question = dec[0].replace("question:", "")
        Question = Question.strip()
        return Question