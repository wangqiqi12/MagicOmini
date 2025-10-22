# NOTE: inference of our new-type(sketch or lineart) data
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# import ipdb;ipdb.set_trace();
import random
from pathlib import Path
import json

import torch
from diffusers.pipelines import FluxPipeline
from PIL import Image
from omini.pipeline.flux_omini import Condition, generate, seed_everything


def infer_1_example(pipe, gt_image_path, cond_image_path, image_prompt, adapter_name, img_size):

    # image = Image.open(gt_image_path)
    cond_image = Image.open(cond_image_path).convert("RGB").resize((img_size, img_size))
    condition = Condition(cond_image, adapter_name)
    pipe.set_adapters(adapter_name)
    seed_everything()
    result_img = generate(
        pipe,
        prompt=image_prompt,
        conditions=[condition],
        width=img_size,
        height=img_size,
    ).images[0]

    concat_image = Image.new("RGB", (img_size*2, img_size))
    # concat_image.paste(image, (0, 0))
    concat_image.paste(condition.condition, (0, 0))
    concat_image.paste(result_img, (img_size*1, 0))
    # concat_image

    output_dir = "test_bug_output"
    os.makedirs(output_dir, exist_ok=True)
    concat_image.save(f"{output_dir}/concat_image_jpg_22_512_2.jpg")

    pass


def sample_json_ids(folder_path, sample_size=5):
    # 获取文件夹内所有文件名
    files = os.listdir(folder_path)
    # 只保留 .json 文件，并提取序号部分（去掉扩展名）
    ids = [os.path.splitext(f)[0] for f in files if f.endswith(".json")]
    # 随机采样 sample_size 个
    sampled = random.sample(ids, sample_size)
    return sampled



def infer_multi_imgs_with_diff_lorasteps(pipe, gt_image_dir, cond_image_dir, jsons_dir, num_imgs, adapter_names, task, img_size):
    # sample 
    gt_image_dir = Path(gt_image_dir)
    cond_image_dir = Path(cond_image_dir)
    jsons_dir = Path(jsons_dir)
    sample_ids = sample_json_ids(jsons_dir, sample_size=num_imgs)
    # NOTE：临时修改为指定id
    sample_ids = ["000018721", "000003157", "000003202", "000003360"]


    print("sample image's ids: ",sample_ids)
    for id in sample_ids:
        # TODO: check
        gt_image_path = gt_image_dir / f"{id}.jpg"
        cond_image_path = cond_image_dir / f"{id}.jpg" # NOTE: 注意这里的格式png！
        json_path = jsons_dir / f"{id}.json"
        # read prompt
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            image_prompt = data.get("prompt", "")
        
        image = Image.open(gt_image_path)
        cond_image = Image.open(cond_image_path).convert("RGB").resize((img_size, img_size))
        

        N_loras = len(adapter_names)
        assert N_loras >= 1
        # start inference!
        results = []
        for adapter_name in adapter_names:
            pipe.set_adapters(adapter_name)
            condition = Condition(cond_image, adapter_name)
            seed_everything()
            result_img = generate(
                pipe,
                prompt=image_prompt,
                conditions=[condition],
                width=img_size,
                height=img_size,
            ).images[0]
            results.append(result_img)

        # concat images
        assert len(results) == N_loras

        # print("saving result imgs")
        # results[0].save("test_output.jpg")
        # import ipdb;ipdb.set_trace();

        concat_image = Image.new("RGB", (img_size * (N_loras + 2), img_size))
        concat_image.paste(image, (0, 0))
        concat_image.paste(condition.condition, (img_size, 0))
        
        for idx, result_img in enumerate(results):
            concat_image.paste(result_img, (img_size * 2 + idx * img_size, 0))
        # concat_image

        output_dir = f"{task}_{img_size}_output"
        os.makedirs(output_dir, exist_ok=True)
        concat_image.save(f"{output_dir}/concat_image_{id}.jpg")
          
    pass

def main():
    
    # NOTE: change task, change to local checkpoint path
    task = "sketch" 
    FLUX_local_path = "/root/private_data/wangqiqi/Omini_ckpts/FLUX.1-dev"
    # lora_local_dir = f"/root/private_data/wangqiqi/MagicOmini/runs/totbs8_1GPU_1024_1024_{task}_AdamW/ckpt"

    lora_local_dir = f"/root/private_data/wangqiqi/MagicOmini/runs/4GPU_bs1_acc8_tot32_1024_1024_r32_sketch_Prodigy/ckpt"
    lora_ckpt_name = "default.safetensors"
    
    img_size = 1024

    gt_image_dir = f"/root/private_data/wangqiqi/data/pexels_{img_size}_{img_size}/{task}/eval/gt"
    cond_image_dir = f"/root/private_data/wangqiqi/data/pexels_{img_size}_{img_size}/{task}/eval/cond_clean"
    jsons_dir = f"/root/private_data/wangqiqi/data/pexels_{img_size}_{img_size}/{task}/eval/jsons"
    num_imgs = 5
    # lora_steps = ["1000", "2000", "5000", "10000", "20000", "50000"]
    # lora_steps = ["10000", "20000", "30000", "40000", "50000"]
    # lora_steps = ["2000","5000","10000","20000","30000"]
    lora_steps = ["15000"]
    # import ipdb;ipdb.set_trace();

    pipe = FluxPipeline.from_pretrained(
        FLUX_local_path, # "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")
    
    adapter_names = []
    for step in lora_steps:
        adapter_name = f"{task}_step{step}"
        pipe.load_lora_weights(
            os.path.join(lora_local_dir, step),
            weight_name=lora_ckpt_name,
            adapter_name=adapter_name,
        )
        adapter_names.append(adapter_name)

    # -------------------- infer 1 example --------------------
    # # NOTE: Remember to align the lora name with the adapter name in Condition(image, adapter_name)!
    # gt_image_path = f"/ssd/zhenlianghe/users/wangqiqi/Working_Now/omini_data/{task}/eval/gt/000002387.jpg"
    # cond_image_path = f"/ssd/zhenlianghe/users/wangqiqi/Working_Now/omini_data/{task}/eval/cond/000002387.jpg"
    # image_prompt = "a white room with a tree painted on the wall"
    

    # -------------------- infer 1 example --------------------

    # # NOTE: temp test
    # gt_image_path = "/root/private_data/wangqiqi/MagicOmini/eval_c2_test.jpg"
    # cond_image_path = "/root/private_data/wangqiqi/MagicOmini/eval_c2_test.jpg"
    # image_prompt = "a green cup of coffee and a pen on a wooden table"
    # adapter_name = adapter_names[0]
    # infer_1_example(pipe, gt_image_path, cond_image_path, image_prompt, adapter_name, img_size)


    # -------------------- infer multi-examples with loras of different steps --------------------
    infer_multi_imgs_with_diff_lorasteps(pipe, gt_image_dir, cond_image_dir, jsons_dir, num_imgs, adapter_names, task, img_size)
    # -------------------- infer multi-examples with loras of different steps --------------------



if __name__ == "__main__":
    main()