import string
import torch
import re

class Preprocessor():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def remove_punctuations(self, text):
        arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
        english_punctuations = string.punctuation
        punctuations_list = arabic_punctuations + english_punctuations
        translator = str.maketrans('', '', punctuations_list)
        return text.translate(translator)

    def remove_repeating_char(self, text):
        return re.sub(r'(.)\1+', r'\1', text)

    def remove_english_characters(self, text):
        return re.sub(r'[a-zA-Z]+', '', text)

    def remove_numbers(self, text):
        return re.sub(r'[1-9]+', '', text)

    def normalize_arabic(self, text):
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)
        return text

    def remove_diacritics(self, text):
        arabic_diacritics = re.compile("""
                                ّ    | # Tashdid
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                            """, re.VERBOSE)
        text = re.sub(arabic_diacritics, '', text)
        return text

    def clean_and_tokenize(self, sentences):
        clean_sentences = []
        for i in range(len(sentences)):
            text = sentences[i]
            #Cleaning
            text = self.remove_punctuations(text)
            text = self.remove_repeating_char(text)
            text = self.remove_english_characters(text)
            text = self.remove_numbers(text)
            #Normalization
            text = self.normalize_arabic(text)
            text = self.remove_diacritics(text)
            #Tokenization
            text = self.tokenizer.tokenize(text)
            if len(text)>510:
                #text = text[:510]
                #text = text[-510:]
                text = text[:255] + text[-255:]
            text.insert(0, self.tokenizer.cls_token)
            text.append(self.tokenizer.sep_token)
            clean_sentences.append(text)
        return clean_sentences

    def creat_tensor(self, sentences):
        max_len = max([len(sen) for sen in sentences])
        batch = len(sentences)
        tensor_data = torch.full((batch, max_len), 0, dtype=torch.int64)
        tensor_mask = torch.zeros(batch, max_len, dtype=torch.int64)
        for i in range(batch):
            ids = self.tokenizer.convert_tokens_to_ids(sentences[i])
            tensor_data[i, 0:len(ids)] = torch.tensor(ids, dtype=torch.int64)
            tensor_mask[i, 0:len(ids)] = torch.ones(1, len(ids), dtype=torch.int64)
        return tensor_data, tensor_mask
