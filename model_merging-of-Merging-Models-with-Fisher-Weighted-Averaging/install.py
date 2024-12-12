import logging
from transformers import AutoModel, AutoTokenizer
import os
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/' #replace with your own location

logging.basicConfig(level=logging.INFO)

# download model
model_name = "sgugger/glue-mrpc"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
