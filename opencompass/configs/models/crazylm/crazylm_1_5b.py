# model configs
from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='CrazyLM-1.5B',
        path="PATH_TO_PRETRAINED_CKPT",
        model_kwargs=dict(device_map='auto', local_files_only=True, trust_remote_code=True),
        max_out_len=256,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1),
        gen_config=dict(temperature=0.2, top_p=0.9, max_new_tokens=1024)
    ),   
]