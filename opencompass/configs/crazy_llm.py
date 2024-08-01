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
datasets = commonsenseqa_datasets + bbh_datasets + gsm8k_datasets + \
            humaneval_datasets + truthfulqa_datasets + chid_datasets  + longbench_datasets

# model configs
from opencompass.models import HuggingFaceBaseModel

# GPUS
NUM_GPUS=2

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='internlm2-chat-1.8b', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/internlm2-chat-1_8b',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='internlm2-chat-7b', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/internlm2-chat-7b',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='Yi-1.5-9B-Chat', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Yi-1.5-9B-Chat',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='Yi-1.5-6B-Chat', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Yi-1.5-6B-Chat',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='Qwen1.5-7B-Chat', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Qwen1.5-7B-Chat',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
        
    dict(
        type=HuggingFaceBaseModel,
        abbr='Qwen2-7B', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Qwen2-7B',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
        
    dict(
        type=HuggingFaceBaseModel,
        abbr='chatglm3-6b', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/chatglm3-6b',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='Nanbeige2-8B-Chat', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Nanbeige2-8B-Chat',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='DCLM-7B', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/DCLM-7B',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='gemma-2-2b', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/gemma-2-2b',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='gemma-2-2b-it', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/gemma-2-2b-it',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='gemma-2-9b', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/gemma-2-9b',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='gemma-2-9b-it', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/gemma-2-9b-it',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    ),
    
    dict(
        type=HuggingFaceBaseModel,
        abbr='Mistral-7B-Instruct-v0.3', # 模型简称，用于结果展示
        path='/mnt/hwfile/optimal/LLMComp-Neurips2024/crazylm/pretrained/Mistral-7B-Instruct-v0.3',
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        max_out_len=256,
        batch_size=8,
        run_cfg=dict(num_gpus=NUM_GPUS),
    )
]

