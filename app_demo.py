import gradio as gr
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

base_model_path = "YOUR_BASE_MODEL_DIR" 
lora_adapter_path = "YOUR_LORA_ADAPTOR__MODEL_PATH"

print("--- 正在加载模型和分词器, 请稍候... ---")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval()

print("--- 模型加载完成！Gradio Web UI 即将启动。 ---")

def chat_stream(message: str, history: list):
    system_prompt = "你是一个优秀的童话故事作家，请根据用户的要求创作一个完整、有教育意义的童话故事。"

    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, ai_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 👑 我的童话故事AI作家 🧸
        
        这是基于 Qwen1.5-1.8B 模型通过 LoRA 微调得到的童话故事生成助手。
        
        **尝试输入一个主题，比如：**
        - 给我讲一个关于“友谊”的故事。
        - 写一个主角是小狐狸，主题是“诚实”的童话。
        - 我想听一个发生在魔法森林里的故事。
        """
    )
    
    gr.ChatInterface(
        fn=chat_stream,
        title="童话故事AI作家",
        examples=[
            ["给我讲一个关于'勇敢'的童话故事"],
            ["写一个关于小松鼠学会'分享'的故事"],
            ["我想听一个关于保护环境的童话"]
        ],
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="请输入你想听的故事主题...", container=False, scale=7),
        clear_btn="清空对话",
        undo_btn="撤销上一轮",
        retry_btn="重新生成",
    )

demo.launch(share=True)