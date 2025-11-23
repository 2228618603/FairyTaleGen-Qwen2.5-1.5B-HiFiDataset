Qwen1.5B Lora Finetuning
● By Team: 孜然与盐组
项目介绍
本项目基于 1.5B 量级大模型（Qwen2.5-1.5B-Instruct），实现 Lora（Low-Rank Adaptation）高效微调，支持个性化任务定制
（如角色对话、领域问答、指令遵循等）。项目提供完整的训练流程、环境配置、一键推理 Demo，无需修改核心代码即可快速
适配自定义数据集。
项目背景
在当前自然语言处理应用中，我们发现通用小模型在生成长文本时存在明显不足，这一问题在童话故事生成任务上表现得尤为突
出。 童话创作本质上是一个具有多重约束的专业化文本生成任务。它首先需要保持逻辑的连贯性，能够构建完整且有吸引力的故
事情节；其次要求语言风格符合文学创作的特点，需要一定的优美度和表现力；同时还要充分考虑低龄读者的认知特点和接受能
力，在词汇选择、句式结构和内容深度上都需做相应调整。 然而，现有通用模型的训练数据主要来自互联网通用文本，与童话故
事的专业要求存在显著差距。这种训练语料与目标文本之间的不匹配，导致模型在多个方面表现欠佳：生成的故事往往缺乏完整
的叙事结构，语言表达过于口语化，难以体现童话文学的特色，同时也经常出现超出儿童理解范围的内容。 具体而言，现有的模
型在童话生成任务中主要面临三个挑战：一是难以把握童话特有的叙事节奏和情节发展规律；二是在语言风格上无法达到儿童文
学应有的生动性和感染力；三是在价值观引导和教育意义融入方面表现得不够自然。 因此，我们希望通过有针对性的微调方法，
让模型更好地学习童话文本的特有模式和创作规律，从而提升其在特定文本类型上的生成能力，满足实际童话创作场景的需求。
这种方法的核心在于弥补通用训练数据与专业领域需求之间的鸿沟，使模型能够产出更符合专业标准的童话作品。
快速开始
训练完成后，直接运行以下命令启动交互式推理 Demo：
python demo.py # for python
python app_demo.py # for gradio demo
详细操作步骤
1. 环境配置
1.1 基础依赖安装
在完成 Python 环境（建议 3.10+）部署后，执行以下命令安装第三方库：
# 升级 pip
python -m pip install --upgrade pip
# 更换清华源加速安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 安装核心依赖
pip install modelscope==1.18.0
pip install transformers==4.44.2
pip install streamlit==1.24.0
pip install sentencepiece==0.2.0
pip install accelerate==0.34.2
pip install datasets==2.20.0
pip install peft==0.11.1
pip install torch>=2.0.0
2. 模型下载
使用 modelscope 下载 1.5B 基础模型，默认保存至 /YOUR_PATH （可自定义路径）。
2.1 下载脚本
在目标路径（如 /root/autodl-tmp ）新建 model_download.py ，粘贴以下代码：
python download_model.py
3. 数据集准备
3.1 数据集格式
微调数据采用「指令-输入-输出」格式，单条样本示例：
{
 "instruction": "扮演[用户填充：角色/任务描述，如「甄嬛」]",
"input": "[用户填充：补充输入，无则留空]",
"output": "[用户填充：预期输出，如「家父是大理寺少卿甄远道」]"
}
3.2 数据集路径
将数据集文件（支持 JSON/CSV 格式）放置在项目根目录的 /dataset 文件夹下，目录结构：
project_root/
├── dataset/
│ └── [YOUR_JSON_PATH]
├── model_download.py
├── app_demo.py
├── qwen1.5B_lora_sft.py
└── qwen_lora_inference.py
4. 训练流程（直接运行即用）
本节提供完整的童话故事创作模型训练代码，核心目标是让模型学会根据用户指令生成**完整、有教育意义**的童话故事，无需
修改核心逻辑，仅需替换占位符即可启动训练。
4.1 训练前准备
1 . 确认已完成「环境配置」（依赖库安装完成）
2 . 下载 1.5B 基础模型（参考前文「模型下载」章节）
3 . 准备童话故事数据集（格式要求见下文）
4.2 数据集格式要求
数据集为 JSON 格式，单条样本包含 instruction （创作要求）、 input （补充信息，无则留空）、 output （标准童话故
事），示例：
[
{
"instruction": "创作一个关于友谊的童话故事，主角是小兔子和小松鼠",
"input": "要求包含遇到困难、互相帮助的情节",
"output": "在茂密的森林里，住着一只活泼的小兔子和一只机灵的小松鼠。一天，小兔子去山上采蘑菇，不小心掉进了一个深坑里...（完
},
{
"instruction": "写一个教育孩子不要挑食的童话故事",
"input": "",
"output": "小熊嘟嘟最讨厌吃蔬菜了，每次吃饭都只吃肉...（完整故事内容）"
}
]
将数据集文件（如 story_train.json ）放置在自定义路径下，后续替换 YOUR_DATASET_PATH 占位符。
4.3 完整训练代码（ qwen1.5B_lora_sft.py ）
打开 qwen1.5B_lora_sft.py ，替换 3 个核心占位符即可运行：
 import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForS
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
def process_func(example):
"""数据预处理函数：将样本格式化为模型可接收的输入"""
MAX_LENGTH = 1024 # 最大序列长度（根据模型能力调整，1.5B 模型建议 ≤1024）
# 固定系统提示词：定义模型角色为「优秀的童话故事作家」
system_prompt = "你是一个优秀的童话故事作家，请根据用户的要求创作一个完整、有教育意义的童话故事。"
# 按照 Qwen 模型 Prompt Template 格式化输入
instruction_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['instru
response_text = f"{example['output']}" # 模型应生成的童话故事
# 编码指令和响应
instruction = tokenizer(instruction_text, add_special_tokens=False)
response = tokenizer(response_text, add_special_tokens=False)
# 拼接输入_ids、注意力掩码、标签（指令部分标签设为 -100，不参与梯度计算）
input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] # eos token 需关注
labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
# 截断过长样本
if len(input_ids) > MAX_LENGTH:
 input_ids = input_ids[:MAX_LENGTH]
attention_mask = attention_mask[:MAX_LENGTH]
labels = labels[:MAX_LENGTH]
return {
"input_ids": input_ids,
"attention_mask": attention_mask,
"labels": labels
}
# -------------------------- 需用户替换的占位符 --------------------------
model_path = "ORIGINAL_MODEL_PATH" # 基础模型路径（如 /root/autodl-tmp/qwen/Qwen2.5-1.5B-Instruct/）
dataset_path = "YOUR_DATASET_PATH(JSON_FILE_FORMAT)" # 数据集路径（如 ./dataset/story_train.json）
final_output_dir = "MODEL_SAFETENSOR_SAVE_PATHS" # 最终模型保存路径（如 ./output/story_lora_final/）
# ------------------------------------------------------------------------
# 加载 Tokenizer（适配 Qwen 系列模型，自动处理特殊token）
tokenizer = AutoTokenizer.from_pretrained(
model_path,
use_fast=False, # 禁用快速Tokenizer，避免格式兼容问题
trust_remote_code=True # 信任自定义模型代码
)
# 若模型无 pad_token，将 eos_token 设为 pad_token（避免训练警告）
if tokenizer.pad_token is None:
tokenizer.pad_token = tokenizer.eos_token
# 加载基础模型（半精度 bf16 加载，节省显存且保证效果）
 model = AutoModelForCausalLM.from_pretrained(
model_path,
device_map="auto", # 自动分配模型到可用设备（GPU/CPU）
torch_dtype=torch.bfloat16 # 半精度计算，需显卡支持（不支持则改为 torch.float16）
)
# 准备模型用于 4/8bit 训练（优化显存占用）
model = prepare_model_for_kbit_training(model)
# 定义 Lora 微调配置（针对童话故事创作任务优化）
lora_config = LoraConfig(
task_type=TaskType.CAUSAL_LM, # 因果语言模型任务（文本生成）
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 训练所
inference_mode=False, # 训练模式（推理时设为 True）
r=8, # Lora 秩（8 为平衡效果和显存的常用值）
lora_alpha=32, # 缩放系数（alpha = r*4，增强信号）
lora_dropout=0.1 # Dropout 比例，防止过拟合
)
# 给模型添加 Lora 适配器（仅训练 Lora 层，冻结基础模型）
model = get_peft_model(model, lora_config)
model.enable_input_require_grads() # 开启输入梯度计算（Lora 训练必需）
model.print_trainable_parameters() # 打印可训练参数比例（通常 <1%，显存占用极低）
# 加载数据集并预处理
ds = load_dataset("json", data_files=dataset_path, split="train") # 加载 JSON 数据集
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names) # 应用预处理函数，删除原始列
 # 定义训练超参数（针对 1.5B 模型和童话故事任务优化）
training_args = TrainingArguments(
output_dir="./output/Qwen_story_lora_100", # 中间模型保存路径
per_device_train_batch_size=2, # 单卡 batch size（1.5B 模型建议 2-4，根据显存调整）
gradient_accumulation_steps=8, # 梯度累加（显存不足时增大，等效提升 batch size）
logging_steps=5, # 每 5 步打印一次训练日志（loss、学习率等）
num_train_epochs=5, # 训练轮数（童话故事数据集建议 3-5 轮，避免过拟合）
save_strategy="epoch", # 按 epoch 保存模型（每训练一轮保存一次）
learning_rate=2e-4, # 学习率（Lora 微调常用 1e-4~2e-4，适配童话故事生成）
gradient_checkpointing=True, # 开启梯度检查点（节省 30%+ 显存）
bf16=True, # 开启 bf16 混合精度训练（加速训练，需显卡支持）
optim="paged_adamw_8bit", # 8bit 优化器（进一步节省显存，不影响效果）
save_total_limit=3 # 最多保存 3 个中间模型（避免占用过多存储空间）
)
# 初始化训练器
trainer = Trainer(
model=model, # 带 Lora 适配器的模型
args=training_args, # 训练超参数
train_dataset=tokenized_ds, # 预处理后的训练集
data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # 自动填充批量数据
)
# 开始训练（首次运行会自动编译模型，后续训练加速）
trainer.train()
 # 保存最终 Lora 权重和 Tokenizer（用于后续推理）
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"训练完成！最终模型已保存至：{final_output_dir}")
4.4 启动训练
1 . 替换代码中 3 个占位符：
● ORIGINAL_MODEL_PATH ：基础模型下载后的路径（如 /root/autodl-tmp/qwen/Qwen2.5-1.5B-Instruct/ ）
● YOUR_DATASET_PATH ：数据集文件路径（如 ./dataset/story_train.json ）
● MODEL_SAFETENSOR_SAVE_PATHS ：最终模型保存路径（如 ./output/story_lora_final/ ）
2 . 执行训练命令：
python qwen1.5B_lora_sft.py
4.5 训练关键说明
● 显存要求：1.5B 模型 + Lora 微调，单卡显存 ≥8GB 即可运行（8GB 显存建议 per_device_train_batch_size=1
+ gradient_accumulation_steps=16 ）
● 日志查看：训练过程中会打印 loss 变化，正常情况下 loss 应逐步下降并趋于稳定
● 模型保存：训练完成后， final_output_dir 路径下会生成 Lora 权重文件
（ adapter_config.json 、 adapter_config.bin ）和 Tokenizer 文件
5. 推理演示（加载 Lora 权重生成童话故事）
训练完成后，通过以下代码加载 Lora 权重，快速生成符合要求的童话故事，支持自定义用户指令。
5.1 完整推理代码（ qwen_lora_inference.py ）
打开 qwen_lora_inference.py ，替换 2 个核心占位符即可运行：
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# -------------------------- 需用户替换的占位符 --------------------------
base_model_path = "YOUR_BASE_MODEL_DIR" # 基础模型路径（与训练时的 model_path 一致）
lora_adapter_path = "YOUR_LORA_ADAPTOR__MODEL_PATH" # Lora 权重保存路径（与训练时的 final_output_dir 一致）
# ------------------------------------------------------------------------
# 加载基础模型的 Tokenizer（需与训练时保持一致）
tokenizer = AutoTokenizer.from_pretrained(
base_model_path,
trust_remote_code=True,
use_fast=False
)
# 加载基础模型（半精度加载，保证推理速度和效果）
base_model = AutoModelForCausalLM.from_pretrained(
base_model_path,
torch_dtype=torch.bfloat16, # 与训练时一致，不支持则改为 torch.float16
device_map="auto", # 自动分配到 GPU/CPU
 trust_remote_code=True
)
# 加载 Lora 适配器（将微调后的权重注入基础模型）
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval() # 切换到推理模式（禁用 Dropout，保证结果稳定）
# 固定系统提示词（与训练时一致，确保模型角色统一）
system_prompt = "你是一个优秀的童话故事作家，请根据用户的要求创作一个完整、有教育意义的童话故事。"
def generate_story(user_prompt):
"""
生成童话故事函数
参数：user_prompt - 用户的创作要求（如「创作一个关于勇敢的童话故事」）
返回：模型生成的完整童话故事
"""
# 构建聊天模板（遵循 Qwen 模型格式，确保与训练数据一致）
messages = [
{"role": "system", "content": system_prompt},
{"role": "user", "content": user_prompt}
]
# 格式化输入文本（自动添加特殊token和对话格式）
text = tokenizer.apply_chat_template(
messages,
tokenize=False,
add_generation_prompt=True # 自动添加 assistant 前缀，触发生成
 )
# 编码输入（转换为模型可识别的张量）
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# 生成童话故事（关闭梯度计算，加快推理速度）
with torch.no_grad():
generated_ids = model.generate(
**model_inputs,
max_new_tokens=1024, # 最大生成长度（童话故事建议 512-1024）
do_sample=True, # 开启采样，增加故事多样性
top_p=0.9, # 核采样比例（0.9 平衡多样性和连贯性）
temperature=0.7, # 温度参数（0.7 避免生成内容过于随机）
repetition_penalty=1.1, # 重复惩罚（减少重复语句）
eos_token_id=tokenizer.eos_token_id # 生成结束符
)
# 截取生成的部分（去除输入prompt）
generated_ids = [
output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids
]
# 解码为自然语言文本
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
return response
# -------------------------- 交互式测试 --------------------------
 if __name__ == "__main__":
print("=== 1.5B 童话故事生成模型 ===")
print("提示：输入创作要求（如「写一个关于环保的童话故事」），输入 'exit' 退出")
while True:
user_input = input("\n请输入你的创作要求：")
if user_input.lower() == "exit":
print("再见！")
break
# 生成并打印故事
story = generate_story(user_input)
print("\n生成的童话故事：")
print("-" * 50)
print(story)
print("-" * 50)
5.2 启动推理
1 . 替换代码中 2 个占位符：
● YOUR_BASE_MODEL_DIR ：基础模型路径（与训练时的 model_path 完全一致）
● YOUR_LORA_ADAPTOR__MODEL_PATH ：Lora 权重保存路径（与训练时的 final_output_dir 完全一致）
2 . 执行推理命令：
python qwen_lora_inference.py
1 . 交互式使用：输入创作要求（如「创作一个关于友谊的童话故事，主角是小猫和小狗」），模型会自动生成完整的童话故
事。
5.3 推理参数调优建议
● 增加故事多样性：提高 temperature （如 0.8-0.9），同时提高 top_p （如 0.95）
● 保证故事连贯性：降低 temperature （如 0.5-0.6），设置 repetition_penalty=1.2
● 延长故事长度：增大 max_new_tokens （如 1536，需注意显存占用）
● 缩短故事长度：减小 max_new_tokens （如 512）
![Uploading image.png…]()
