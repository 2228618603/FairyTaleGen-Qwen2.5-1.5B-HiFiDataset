import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('qwen/Qwen2.5-1.5B-Instruct', cache_dir='/root/autodl-tmp/users/djl/llm', revision='master')
