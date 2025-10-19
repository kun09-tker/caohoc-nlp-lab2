from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer

class ChatbotModel:
    def __init__(self, model_name, checkpoint_dir):
        # Load lại base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        # Load lại LoRA adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model.enable_adapter_layers()
        print("✅ Loaded LoRA adapter từ:", checkpoint_dir)
        self.model = model
        self.tokenizer = tokenizer

    def generate_ans(self, question):
        inputs = self.tokenizer(question, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=32)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

