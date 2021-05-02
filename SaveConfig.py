from transformers import DataCollatorForLanguageModeling, RobertaConfig, ReformerConfig, XLNetConfig, XLMConfig, XLMRobertaConfig
from transformers import AdamW, RobertaModel, AutoModel, RobertaTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM, XLNetLMHeadModel, ReformerForMaskedLM, ReformerForQuestionAnswering
config = ReformerConfig.from_pretrained("google/reformer-enwik8")
config.save_pretrained("/project/6033386/partha9/model_cache/reformer_config")
