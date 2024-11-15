from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time

from langchain_community.llms import huggingface_pipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import huggingface_hub
from langchain.chains import retrieval_qa
from langchain.vectorstores import chroma


model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

device = f'cude:{cuda.current_device()}' if cuda.is_available() else 'cpu'

#set quantization config to load large model with less GPU memory
#this requires 'bitsandbytes' library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

time_1 = time()
model_config = transformers.AutoConfig.from_pretrained(
    model_id
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
time_2 = time()
print(f"Prepare model, tokenizer: {round(time_2-time_1, 3)} sec.")

time_1 = time()
query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
time_2 = time()
print(f"Prepare pipeline: {round(time_2-time_1, 3)} sec.")

def test_model(tokenizer, pipeline, prompt):
    time_1 = time()
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_toke_id,
        max_length=200,)
    time_2=time()
    print(f"Test inference: {round(time_2-time_1, 3)} sec.")
    for seq in sequences:
        print(f"Result: {seq['generated_test']}")


print(test_model(tokenizer, query_pipeline, "Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words"))
#llm = huggingface_pipeline(pipeline=query_pipeline)
#llm(prompt="Please")
