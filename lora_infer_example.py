# NOTE: inpainting lora example, using ominicontrol original code mostly

import os

# os.chdir("..")

import torch
from diffusers.pipelines import FluxPipeline
from PIL import Image

from omini.pipeline.flux_omini import Condition, generate, seed_everything

# NOTE: change to local checkpoint path
local_path = "/root/private_data/wangqiqi/Omini_ckpts/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(
    local_path, # "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
# NOTE: mind ckpt path!
local_path = "/root/private_data/wangqiqi/MagicOmini/runs/4GPU_bs1_acc8_tot32_1024_1024_r32_sketch_Prodigy/ckpt/15000"
# local_path = "/root/private_data/wangqiqi/Omini_ckpts/lora_sketch" # 改成512*512就没有问题
pipe.load_lora_weights(
    local_path,
    weight_name=f"default.safetensors",
    adapter_name="sketch",
)

img_size = 1024 # 改成512就没有问题
pipe.set_adapters("sketch")

# -------------------- example 1 --------------------

image = Image.open("./test4.jpg").convert("RGB").resize((img_size, img_size))

# image.save("example_out_1.jpg")
# masked_image = image.copy()
# masked_image.paste((0, 0, 0), (128, 100, 384, 220))

condition = Condition(image, "sketch")

seed_everything()
result_img = generate(
    pipe,
    prompt="apple and lemon",
    conditions=[condition],
    width=img_size,
    height=img_size,
).images[0]

result_img.save(f"example_out_{img_size}_test4.jpg")

# concat_image = Image.new("RGB", (1536, 512))
# concat_image.paste(image, (0, 0))
# concat_image.paste(condition.condition, (512, 0))
# concat_image.paste(result_img, (1024, 0))
# # concat_image

# output_dir = "lora_example_output"
# os.makedirs(output_dir, exist_ok=True)
# concat_image.save(f"{output_dir}/concat_image_vase_2.jpg")
