import time

from googletrans import Translator

class translate:
    def __init__(self):
        self.translator = Translator()

    def translate_en_vi(self, text):
        return self.translator.translate(text, src='en', dest='vi').text

    def translate_vi_en(self, text):
        return self.translator.translate(text, src='vi', dest='en').text