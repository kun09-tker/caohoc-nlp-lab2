from deep_translator import GoogleTranslator

class translate:
    def __init__(self):
        self.translator_vi = GoogleTranslator(source='auto', target='vi')
        self.translator_eng = GoogleTranslator(source='auto', target='en')

    def translate_en_vi(self, text):
        return self.translator_vi.translate(text)

    def translate_vi_en(self, text):
        return self.translator_eng.translate(text)