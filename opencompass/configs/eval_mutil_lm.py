from mmengine.config import read_base

with read_base():
    from .lark import lark_bot_url
    from .datasets.commonsenseqa.commonsenseqa_gen import commonsenseqa_datasets
    from .datasets.bbh.bbh_gen import bbh_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.longbench.longbench import longbench_datasets
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets
    from .datasets.FewCLUE_chid.FewCLUE_chid_gen import chid_datasets

# dataset configs
datasets = commonsenseqa_datasets  + gsm8k_datasets + \
            humaneval_datasets + truthfulqa_datasets + chid_datasets

# large benchmark
# datasets = longbench_datasets
# datasets = bbh_datasets

# model configs
from opencompass.models import HuggingFaceBaseModel
from transformers import AutoConfig

# GPUS
NUM_GPUS=1
BATCH_SIZE=2

models = [
    # dict(
    
    # dict(
    #     type=HuggingFaceBaseModel,
    #     abbr='gemma-2-2b', # 模型简称，用于结果展示
    #     path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/gemma-2-2b',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     max_out_len=256,
    #     batch_size=BATCH_SIZE,
    #     run_cfg=dict(num_gpus=NUM_GPUS)
    # ),
    
    # dict(
    #     type=HuggingFaceBaseModel,
    #     abbr='chatglm3-6b', # 模型简称，用于结果展示
    #     path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/chatglm3-6b',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     max_out_len=256,
    #     batch_size=BATCH_SIZE,
    #     run_cfg=dict(num_gpus=NUM_GPUS)
    # ),
    #     type=HuggingFaceBaseModel,
    #     abbr='gemma-2-2b-it', # 模型简称，用于结果展示
    #     path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/gemma-2-2b-it',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     max_out_len=256,
    #     batch_size=BATCH_SIZE,
    #     run_cfg=dict(num_gpus=NUM_GPUS)
    # ),
    
    # dict(
    #     type=HuggingFaceBaseModel,
    #     abbr='internlm2-chat-1_8b', # 模型简称，用于结果展示
    #     path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/internlm2-chat-1_8b',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     max_out_len=256,
    #     batch_size=BATCH_SIZE,
    #     run_cfg=dict(num_gpus=NUM_GPUS)
    # ),
    
    # dict(
    #     type=HuggingFaceBaseModel,
    #     abbr='Phi-3-mini-4k-instruct', # 模型简称，用于结果展示
    #     path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Phi-3-mini-4k-instruct',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     max_out_len=256,
    #     batch_size=BATCH_SIZE,
    #     run_cfg=dict(num_gpus=NUM_GPUS)
    # ),
    
    # dict(
    #     type=HuggingFaceBaseModel,
    #     abbr='Phi-3-mini-128k-instruct', # 模型简称，用于结果展示
    #     path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Phi-3-mini-128k-instruct',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     max_out_len=256,
    #     batch_size=BATCH_SIZE,
    #     run_cfg=dict(num_gpus=NUM_GPUS)
    # ),
    
    # dict(
    #     type=HuggingFaceBaseModel,
    #     abbr='Qwen2-0.5B', # 模型简称，用于结果展示
    #     path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Qwen2-0.5B',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     max_out_len=256,
    #     batch_size=BATCH_SIZE,
    #     run_cfg=dict(num_gpus=NUM_GPUS)
    # ),
    
    # dict(
    #     type=HuggingFaceBaseModel,
    #     abbr='Qwen2-0.5B-Instruct', # 模型简称，用于结果展示
    #     path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Qwen2-0.5B-Instruct',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     max_out_len=256,
    #     batch_size=BATCH_SIZE,
    #     run_cfg=dict(num_gpus=NUM_GPUS)
    # ),
    
    # dict(
    #     type=HuggingFaceBaseModel,
    #     abbr='Qwen2-1.5B', # 模型简称，用于结果展示
    #     path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Qwen2-1.5B',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     max_out_len=256,
    #     batch_size=BATCH_SIZE,
    #     run_cfg=dict(num_gpus=NUM_GPUS)
    # ),
    
    # dict(
    #     type=HuggingFaceBaseModel,
    #     abbr='Qwen2-1.5B-Instruct', # 模型简称，用于结果展示
    #     path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Qwen2-1.5B-Instruct',
    #     model_kwargs=dict(device_map='auto', trust_remote_code=True),
    #     max_out_len=256,
    #     batch_size=BATCH_SIZE,
    #     run_cfg=dict(num_gpus=NUM_GPUS)
    # ),

    dict(
        type=HuggingFaceBaseModel,
        abbr='CrazyLM-1.5B', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/outputs/crazylm_v1_0_pretrain/2024-08-15/11-32-56_c4_default/checkpoint-9000',
        model_kwargs=dict(device_map='auto', local_files_only=True),
        max_out_len=256,
        batch_size=BATCH_SIZE,
        run_cfg=dict(num_gpus=NUM_GPUS)
    ),
    
]

