[**🇨🇳中文**](https://github.com/shibing624/MedicalGPT/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/MedicalGPT/blob/main/README_EN.md) | [**📖文档/Docs**](https://github.com/shibing624/MedicalGPT/wiki) | [**🤖模型/Models**](https://huggingface.co/shibing624)

-----------------

# MedicalGPT: Training Medical GPT Model
[![HF Models](https://img.shields.io/badge/Hugging%20Face-shibing624-green)](https://huggingface.co/shibing624)
[![Github Stars](https://img.shields.io/github/stars/shibing624/MedicalGPT?color=yellow)](https://star-history.com/#shibing624/MedicalGPT&Timeline)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)

## 📖 Introduction

**MedicalGPT** training medical GPT model with ChatGPT training pipeline, implemantation of Pretraining,
Supervised Finetuning, RLHF(Reward Modeling and Reinforcement Learning) and DPO(Direct Preference Optimization).

**MedicalGPT** 训练医疗大模型，实现了包括增量预训练、有监督微调、RLHF(奖励建模、强化学习训练)和DPO(直接偏好优化)。

<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/dpo.jpg" width="860" />


## 😊 Features


基于ChatGPT Training Pipeline，本项目实现了领域模型--医疗行业语言大模型的训练：


- 第一阶段：PT(Continue PreTraining)增量预训练，在海量领域文档数据上二次预训练GPT模型，以适应领域数据分布（可选）
- 第二阶段：SFT(Supervised Fine-tuning)有监督微调，构造指令微调数据集，在预训练模型基础上做指令精调，以对齐指令意图，并注入领域知识
- 第三阶段
  - RLHF(Reinforcement Learning from Human Feedback)基于人类反馈对语言模型进行强化学习，分为两步：
    - RM(Reward Model)奖励模型建模，构造人类偏好排序数据集，训练奖励模型，用来建模人类偏好，主要是"HHH"原则，具体是"helpful, honest, harmless"
    - RL(Reinforcement Learning)强化学习，用奖励模型来训练SFT模型，生成模型使用奖励或惩罚来更新其策略，以便生成更高质量、更符合人类偏好的文本
  - [DPO(Direct Preference Optimization)](https://arxiv.org/pdf/2305.18290.pdf)直接偏好优化方法，DPO通过直接优化语言模型来实现对其行为的精确控制，而无需使用复杂的强化学习，也可以有效学习到人类偏好，DPO相较于RLHF更容易实现且易于训练，效果更好
  - [ORPO](https://arxiv.org/abs/2403.07691)比值比偏好优化，不需要参考模型（ref_model）的优化方法，通过ORPO，LLM可以同时学习SFT和对齐，将两个过程整合为单一步骤，缓解模型灾难性遗忘问题


### Release Models


| Model                                                                                                             | Base Model                                                                              | Introduction                                                                                                                                                                 |
|:------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [shibing624/ziya-llama-13b-medical-lora](https://huggingface.co/shibing624/ziya-llama-13b-medical-lora)           | [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)       | 在240万条中英文医疗数据集[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)上SFT微调了一版Ziya-LLaMA-13B模型，医疗问答效果有提升，发布微调后的LoRA权重(单轮对话)                                 |
| [shibing624/ziya-llama-13b-medical-merged](https://huggingface.co/shibing624/ziya-llama-13b-medical-merged)       | [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)       | 在240万条中英文医疗数据集[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)上SFT微调了一版Ziya-LLaMA-13B模型，医疗问答效果有提升，发布微调后的完整模型权重(单轮对话)                                 |
| [shibing624/vicuna-baichuan-13b-chat-lora](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat-lora)       | [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | 在10万条多语言ShareGPT GPT4多轮对话数据集[shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) 和 医疗数据集[shibing624/medical](https://huggingface.co/datasets/shibing624/medical) 上SFT微调了一版baichuan-13b-chat多轮问答模型，日常问答和医疗问答效果有提升，发布微调后的LoRA权重 |
| [shibing624/vicuna-baichuan-13b-chat](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat)                 | [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | 在10万条多语言ShareGPT GPT4多轮对话数据集[shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) 和 医疗数据集[shibing624/medical](https://huggingface.co/datasets/shibing624/medical) 上SFT微调了一版baichuan-13b-chat多轮问答模型，日常问答和医疗问答效果有提升，发布微调后的完整模型权重 |
| [shibing624/llama-3-8b-instruct-262k-chinese](https://huggingface.co/shibing624/llama-3-8b-instruct-262k-chinese) | [Llama-3-8B-Instruct-262k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k)  | 在2万条中英文偏好数据集[shibing624/DPO-En-Zh-20k-Preference](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference)上使用ORPO方法微调得到的超长文本多轮对话模型，适用于RAG、多轮对话                   |

演示[shibing624/vicuna-baichuan-13b-chat](https://huggingface.co/shibing624/vicuna-baichuan-13b-chat)模型效果：
<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/demo-screen.gif" width="860" />
具体case见[Inference Examples](#inference-examples)

## ▶️ Demo


我们提供了一个简洁的基于gradio的交互式web界面，启动服务后，可通过浏览器访问，输入问题，模型会返回答案。

启动服务，命令如下：
```shell
CUDA_VISIBLE_DEVICES=0 python gradio_demo.py --base_model path_to_llama_hf_dir --lora_model path_to_lora_dir
```

参数说明：

- `--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录，也可使用HF Model Hub模型调用名称
- `--lora_model {lora_model}`：LoRA文件所在目录，也可使用HF Model Hub模型调用名称。若lora权重已经合并到预训练模型，则删除--lora_model参数
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--base_model相同
- `--template_name`：模板名称，如`vicuna`、`alpaca`等。若不提供此参数，则其默认值是vicuna
- `--only_cpu`: 仅使用CPU进行推理
- `--resize_emb`：是否调整embedding大小，若不调整，则使用预训练模型的embedding大小，默认不调整


## 💾 Install
#### Updating the requirements
`requirements.txt`会不时更新以适配最新功能，使用以下命令更新依赖:

```markdown
git clone https://github.com/shibing624/MedicalGPT
cd MedicalGPT
pip install -r requirements.txt --upgrade
```

#### Hardware Requirement (显存/VRAM)


\* *估算值*

| 训练方法  | 精度          |   7B  |  13B  |  30B  |   70B  |  110B  |  8x7B |  8x22B |
|-------|-------------| ----- | ----- | ----- | ------ | ------ | ----- | ------ |
| 全参数   | AMP(自动混合精度) | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| 全参数   | 16          |  60GB | 120GB | 300GB |  600GB |  900GB | 400GB | 1200GB |
| LoRA  | 16          |  16GB |  32GB |  64GB |  160GB |  240GB | 120GB |  320GB |
| QLoRA | 8           |  10GB |  20GB |  40GB |   80GB |  140GB |  60GB |  160GB |
| QLoRA | 4           |   6GB |  12GB |  24GB |   48GB |   72GB |  30GB |   96GB |
| QLoRA | 2           |   4GB |   8GB |  16GB |   24GB |   48GB |  18GB |   48GB |

## 🚀 Training Pipeline

Training Stage:

| Stage                          | Introduction | Python script                                                                                           | Shell script                                                                  |
|:-------------------------------|:-------------|:--------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|
| Continue Pretraining           | 增量预训练        | [pretraining.py](https://github.com/shibing624/MedicalGPT/blob/main/pretraining.py)                     | [run_pt.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_pt.sh)     |
| Supervised Fine-tuning         | 有监督微调        | [supervised_finetuning.py](https://github.com/shibing624/MedicalGPT/blob/main/supervised_finetuning.py) | [run_sft.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_sft.sh)   |
| Direct Preference Optimization | 直接偏好优化       | [dpo_training.py](https://github.com/shibing624/MedicalGPT/blob/main/dpo_training.py)                   | [run_dpo.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_dpo.sh)   |
| Reward Modeling                | 奖励模型建模       | [reward_modeling.py](https://github.com/shibing624/MedicalGPT/blob/main/reward_modeling.py)             | [run_rm.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_rm.sh)     |
| Reinforcement Learning         | 强化学习         | [ppo_training.py](https://github.com/shibing624/MedicalGPT/blob/main/ppo_training.py)                   | [run_ppo.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_ppo.sh)   |
| ORPO                           | 概率偏好优化       | [orpo_training.py](https://github.com/shibing624/MedicalGPT/blob/main/orpo_training.py)                  | [run_orpo.sh](https://github.com/shibing624/MedicalGPT/blob/main/run_orpo.sh) |

- 提供完整PT+SFT+DPO全阶段串起来训练的pipeline：[run_training_dpo_pipeline.ipynb](https://github.com/shibing624/MedicalGPT/blob/main/run_training_dpo_pipeline.ipynb) ，其对应的colab： [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/run_training_dpo_pipeline.ipynb)，运行完大概需要15分钟
- 提供完整PT+SFT+RLHF全阶段串起来训练的pipeline：[run_training_ppo_pipeline.ipynb](https://github.com/shibing624/MedicalGPT/blob/main/run_training_ppo_pipeline.ipynb) ，其对应的colab： [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/run_training_ppo_pipeline.ipynb) ，运行完大概需要20分钟
- 提供基于知识库文件的LLM问答功能（RAG）：[chatpdf.py](https://github.com/shibing624/MedicalGPT/blob/main/chatpdf.py)
- [训练参数说明](https://github.com/shibing624/MedicalGPT/blob/main/docs/training_params.md) | [训练参数说明wiki](https://github.com/shibing624/MedicalGPT/wiki/%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)
- [数据集](https://github.com/shibing624/MedicalGPT/blob/main/docs/datasets.md) | [数据集wiki](https://github.com/shibing624/MedicalGPT/wiki/%E6%95%B0%E6%8D%AE%E9%9B%86)
- [扩充词表](https://github.com/shibing624/MedicalGPT/blob/main/docs/extend_vocab.md) | [扩充词表wiki](https://github.com/shibing624/MedicalGPT/wiki/%E6%89%A9%E5%85%85%E4%B8%AD%E6%96%87%E8%AF%8D%E8%A1%A8)
- [FAQ](https://github.com/shibing624/MedicalGPT/blob/main/docs/FAQ.md) | [FAQ_wiki](https://github.com/shibing624/MedicalGPT/wiki/FAQ)

#### Supported Models

| Model Name                                                           | Model Size                    | Target Modules  | Template  |
|----------------------------------------------------------------------|-------------------------------|-----------------|-----------|
| [Baichuan](https://github.com/baichuan-inc/baichuan-13B)             | 7B/13B                        | W_pack          | baichuan  |
| [Baichuan2](https://github.com/baichuan-inc/Baichuan2)               | 7B/13B                        | W_pack          | baichuan2 |
| [BLOOMZ](https://huggingface.co/bigscience/bloomz)                   | 560M/1.1B/1.7B/3B/7.1B/176B   | query_key_value | vicuna    |
| [ChatGLM](https://github.com/THUDM/ChatGLM-6B)                       | 6B                            | query_key_value | chatglm   |
| [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)                     | 6B                            | query_key_value | chatglm2  |
| [ChatGLM3](https://github.com/THUDM/ChatGLM3)                        | 6B                            | query_key_value | chatglm3  |
| [Cohere](https://huggingface.co/CohereForAI/c4ai-command-r-plus)     | 104B                          | q_proj,v_proj   | cohere    |
| [DeepSeek](https://github.com/deepseek-ai/DeepSeek-LLM)              | 7B/16B/67B                    | q_proj,v_proj   | deepseek  |
| [DeepSeek3](https://github.com/deepseek-ai/DeepSeek-V3)              | 671B                         | q_proj,v_proj   | deepseek3 |
| [InternLM2](https://github.com/InternLM/InternLM)                    | 7B/20B                        | wqkv            | intern2   |
| [LLaMA](https://github.com/facebookresearch/llama)                   | 7B/13B/33B/65B                | q_proj,v_proj   | alpaca    |
| [LLaMA2](https://huggingface.co/meta-llama)                          | 7B/13B/70B                    | q_proj,v_proj   | llama2    |
| [LLaMA3](https://huggingface.co/meta-llama)                          | 8B/70B                        | q_proj,v_proj   | llama3    |
| [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | 7B/8x7B                       | q_proj,v_proj   | mistral   |
| [Orion](https://github.com/OrionStarAI/Orion)                        | 14B                           | q_proj,v_proj   | orion     |
| [Qwen](https://github.com/QwenLM/Qwen)                               | 1.8B/7B/14B/72B               | c_attn          | qwen      |
| [Qwen1.5](https://huggingface.co/Qwen/Qwen1.5-72B)                   | 0.5B/1.8B/4B/14B/32B/72B/110B | q_proj,v_proj   | qwen      |
| [Qwen2](https://github.com/QwenLM/Qwen2)                             | 0.5B/1.5B/7B/72B              | q_proj,v_proj   | qwen      |
| [Qwen2.5](https://github.com/QwenLM/Qwen2.5)                         | 0.5B/1.8B/4B/14B/72B        | q_proj,v_proj   | qwen      |
| [XVERSE](https://github.com/xverse-ai/XVERSE-13B)                    | 13B                           | query_key_value | xverse    |
| [Yi](https://github.com/01-ai/Yi)                                    | 6B/34B                        | q_proj,v_proj   | yi        |




## 💻 Inference
训练完成后，现在我们加载训练好的模型，验证模型生成文本的效果。

```shell
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model path_to_model_hf_dir \
    --lora_model path_to_lora \
    --interactive
```

参数说明：

- `--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录
- `--tokenizer_path {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录
- `--lora_model {lora_model}`：LoRA解压后文件所在目录，也可使用HF Model Hub模型调用名称。如果已经合并了LoRA权重到预训练模型，则可以不提供此参数
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--base_model相同
- `--template_name`：模板名称，如`vicuna`、`alpaca`等。若不提供此参数，则其默认值是vicuna
- `--interactive`：以交互方式启动多轮问答，使用流式推理
- `--data_file {file_name}`：非交互方式启动下，读取file_name中的的内容进行batch预测
- `--output_file {file_name}`：非交互式方式下，将预测的结果以jsonl格式写入file_name
- `--resize_emb`：是否调整embedding大小，若不调整，则使用预训练模型的embedding大小，默认不调整
- `--only_cpu`：仅使用CPU进行推理
- `--gpus {gpu_ids}`：指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2



## 📚 Dataset
### 医疗数据集

- 240万条中文医疗数据集(包括预训练、指令微调和奖励数据集)：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- 22万条中文医疗对话数据集(华佗项目)：[shibing624/huatuo_medical_qa_sharegpt](https://huggingface.co/datasets/shibing624/huatuo_medical_qa_sharegpt) 【本项目支持格式】



