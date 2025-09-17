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

pipe.load_lora_weights(
    "Yuanshi/OminiControl",
    weight_name=f"experimental/fill.safetensors",
    adapter_name="fill",
)

# -------------------- example 1 --------------------

image = Image.open("assets/monalisa.jpg").convert("RGB").resize((512, 512))

masked_image = image.copy()
masked_image.paste((0, 0, 0), (128, 100, 384, 220))

condition = Condition(masked_image, "fill")

seed_everything()
result_img = generate(
    pipe,
    prompt="The Mona Lisa is wearing a white VR headset with 'Omini' written on it.",
    conditions=[condition],
).images[0]

concat_image = Image.new("RGB", (1536, 512))
concat_image.paste(image, (0, 0))
concat_image.paste(condition.condition, (512, 0))
concat_image.paste(result_img, (1024, 0))
# concat_image

output_dir = "example_output"
os.makedirs(output_dir, exist_ok=True)
concat_image.save(f"{output_dir}/concat_image_1.jpg")

# -------------------- example 2 --------------------

image = Image.open("assets/book.jpg").convert("RGB").resize((512, 512))

w, h, min_dim = image.size + (min(image.size),)
image = image.crop(
    ((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2)
).resize((512, 512))


masked_image = image.copy()
masked_image.paste((0, 0, 0), (150, 150, 350, 250))
masked_image.paste((0, 0, 0), (200, 380, 320, 420))

condition = Condition(masked_image, "fill")

seed_everything()
result_img = generate(
    pipe,
    prompt="A yellow book with the word 'OMINI' in large font on the cover. The text 'for FLUX' appears at the bottom.",
    conditions=[condition],
).images[0]

concat_image = Image.new("RGB", (1536, 512))
concat_image.paste(image, (0, 0))
concat_image.paste(condition.condition, (512, 0))
concat_image.paste(result_img, (1024, 0))

concat_image.save(f"{output_dir}/concat_image_2.jpg")