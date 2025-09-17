# NOTE: inpainting lora example, using ominicontrol original code mostly

import os

# os.chdir("..")

import torch
from diffusers.pipelines import FluxPipeline
from PIL import Image

from omini.pipeline.flux_omini import Condition, generate, seed_everything

# NOTE: change to local checkpoint path
local_path = "/ssd/zhenlianghe/users/wangqiqi/Working_Now/omini_ckpts/FLUX_ckpts/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(
    local_path, # "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
# NOTE: mind ckpt path!
local_path = "/ssd/zhenlianghe/users/wangqiqi/Working_Now/OminiControl/runs/20250901-160703/ckpt/10000"
pipe.load_lora_weights(
    local_path,
    weight_name=f"default.safetensors",
    adapter_name="fill",
)

# -------------------- example 1 --------------------

image = Image.open("assets/vase.jpg").convert("RGB").resize((512, 512))

masked_image = image.copy()
masked_image.paste((0, 0, 0), (128, 100, 384, 220))

condition = Condition(masked_image, "fill")

seed_everything()
result_img = generate(
    pipe,
    prompt="Vase.",
    conditions=[condition],
).images[0]

concat_image = Image.new("RGB", (1536, 512))
concat_image.paste(image, (0, 0))
concat_image.paste(condition.condition, (512, 0))
concat_image.paste(result_img, (1024, 0))
# concat_image

output_dir = "lora_example_output"
os.makedirs(output_dir, exist_ok=True)
concat_image.save(f"{output_dir}/concat_image_vase_2.jpg")
