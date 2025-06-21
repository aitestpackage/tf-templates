from transformers import AutoTokenizer, BertForSequenceClassification, BertForMaskedLM, BertForQuestionAnswering, BertForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
# bert_model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=6)

from transformers import AutoTokenizer, ModernBertForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
modern_bert_model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=6)

from transformers import AutoTokenizer, T5ForSequenceClassification, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
T5_model = T5ForSequenceClassification.from_pretrained("google-t5/t5-base", num_labels=6)
# T5_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

from transformers import AutoTokenizer, MobileBertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
mobile_bert_model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=6)