
import torch

from backend.aspect_sentiment.LLM_CL.LLM_CL import LLM_CL
from collections import Counter

class AspectSentimentClassification:
    def __init__(self, checkpoint_path, tokenizer_path):
        self.mapping = {1: "neutral", 2: "positive", 0: "negative"}
        self.domain_names = ['rest', 'laptop', 'Speaker', 'Router', 'Computer', 'Nokia6610', 'NikonCoolpix4300', 'CreativeLabsNomadJukeboxZenXtra40GB', 'CanonG3', 'ApexAD2600Progressive', 'CanonPowerShotSD500', 'CanonS100', 'DiaperChamp', 'HitachiRouter', 'ipod', 'LinksysRouter', 'MicroMP3', 'Nokia6600', 'Norton']
        self.llcm_model = self.load_model(checkpoint_path, tokenizer_path)

    def load_model(self, checkpoint_path, tokenizer_path):
        llm_cl = LLM_CL(self.domain_names, tokenizer_path, rank_domain=16, alpha_domain=32,
                        device="cpu", rank_share=16, alpha_share=32)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
        llm_cl.load_state_dict(checkpoint['model_state_dict'])
        BEST_F1 = checkpoint['best_f1']
        # print(checkpoint['covariance'])
        # print(checkpoint['domain_prototypes'])
        print(f'Loaded model {checkpoint_path} with F1 = {BEST_F1:.4f} successfuly')
        return llm_cl

    def get_input_sep(self, review, aspect):
        return f"{review} [SEP] {aspect}"

    def predict_sentiment(self, review, aspect):
        text = self.get_input_sep(review, aspect)
        tokenized_input = self.llcm_model.tokenizer(text, max_length=512, return_tensors='pt', \
                                                    truncation=True, padding=True).to(self.llcm_model.model.device)
        for d in self.domain_names:
            vottinglist = []
            output = self.llcm_model.model(**tokenized_input, domain_name=d)
            logits = output.logits
            predicted_class = self.mapping[torch.argmax(logits, dim=1).item()]
            vottinglist.append(predicted_class)
        counts = Counter(vottinglist)
        most_common_emotion, _freq = counts.most_common(1)[0]
        return most_common_emotion