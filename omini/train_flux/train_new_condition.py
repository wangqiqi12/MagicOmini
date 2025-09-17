import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import numpy as np

from PIL import Image, ImageDraw

from datasets import load_dataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, convert_to_condition, generate

import json
import glob
from pathlib import Path


class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size=(512, 512),
        target_size=(512, 512),
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __get_condition__(self, image, condition_type, idx):
        condition_size = self.condition_size
        position_delta = np.array([0, 0])
        if condition_type in ["canny", "coloring", "deblurring", "depth"]:
            image, kwargs = image.resize(condition_size), {}
            if condition_type == "deblurring":
                blur_radius = random.randint(1, 10)
                kwargs["blur_radius"] = blur_radius
            condition_img = convert_to_condition(condition_type, image, **kwargs)
        elif condition_type == "depth_pred":
            depth_img = convert_to_condition("depth", image)
            condition_img = image.resize(condition_size)
            image = depth_img.resize(condition_size)
        elif condition_type == "fill":
            condition_img = image.resize(condition_size).convert("RGB")
            w, h = image.size
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            if random.random() > 0.5:
                mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            )
        elif condition_type == "sr":
            condition_img = image.resize(condition_size)
            position_delta = np.array([0, -condition_size[0] // 16])
        elif condition_type == "lineart" or condition_type == "sketch":
            condition_img = self.base_dataset[idx]['condition_img']
        else:
            raise ValueError(f"Condition type {condition_type} is not  implemented.")
        return condition_img, position_delta

    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize(self.target_size).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]

        condition_size = self.condition_size
        position_scale = self.position_scale

        condition_img, position_delta = self.__get_condition__(
            image, self.condition_type, idx
        )

        # Randomly drop text or image (for training)
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob

        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new("RGB", condition_size, (0, 0, 0))

        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(condition_img),
            "condition_type_0": self.condition_type,
            "position_delta_0": position_delta,
            "description": description,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }

# NOTE: 新建一个CustomImageConditionDataset，
# 专门用来处理不是HF格式的数据集的加载情况
class CustomImageConditionDataset(Dataset):
    """
    自定义图像条件数据集，支持分离的文件夹结构
    数据格式要求：
    - data_dir/gt/ - 原图文件夹
    - data_dir/cond/ - 条件图文件夹  
    - data_dir/jsons/ - JSON元数据文件夹
    - JSON格式: {"prompt": "描述文本"}
    """

    def __init__(
        self,
        data_dir,  # 数据集根目录路径
        condition_size=(512, 512),
        target_size=(512, 512),
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.data_dir = Path(data_dir)
        self.gt_dir = self.data_dir / "gt"
        self.cond_dir = self.data_dir / "cond"
        self.jsons_dir = self.data_dir / "jsons"
        
        # 验证文件夹存在
        if not self.gt_dir.exists():
            raise ValueError(f"GT folder not found: {self.gt_dir}")
        if not self.cond_dir.exists():
            raise ValueError(f"Condition folder not found: {self.cond_dir}")
        if not self.jsons_dir.exists():
            raise ValueError(f"JSONs folder not found: {self.jsons_dir}")
            
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

        # 扫描JSON文件，以此为基准构建样本列表
        json_extensions = ["*.json"]
        json_files = []
        for ext in json_extensions:
            json_files.extend(glob.glob(str(self.jsons_dir / ext)))

        json_files = sorted(json_files)
        print(f"Found {len(json_files)} JSON files in {self.jsons_dir}")

        # 构建有效样本列表，确保GT图像、条件图像和JSON三者都存在
        self.valid_samples = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for json_path in json_files:
            json_filename = Path(json_path).stem  # 获取文件名（不含扩展名）
            
            # 查找对应的GT图像
            gt_img_path = None
            for ext in image_extensions:
                candidate_path = self.gt_dir / f"{json_filename}{ext}"
                if candidate_path.exists():
                    gt_img_path = str(candidate_path)
                    break
            
            # 查找对应的条件图像
            cond_img_path = None
            for ext in image_extensions:
                candidate_path = self.cond_dir / f"{json_filename}{ext}"
                if candidate_path.exists():
                    cond_img_path = str(candidate_path)
                    break
            
            # 只有三个文件都存在才添加到有效样本
            if gt_img_path and cond_img_path and os.path.exists(json_path):
                self.valid_samples.append({
                    "gt_img_path": gt_img_path,
                    "cond_img_path": cond_img_path,
                    "json_path": json_path,
                    "key": json_filename,
                    "url": str(self.data_dir)
                })
            else:
                missing = []
                if not gt_img_path:
                    missing.append("GT image")
                if not cond_img_path:
                    missing.append("condition image")
                if not os.path.exists(json_path):
                    missing.append("JSON file")
                print(f"Warning: Missing {', '.join(missing)} for {json_filename}")

        print(f"Valid samples: {len(self.valid_samples)}")
        print(f"GT folder: {self.gt_dir}")
        print(f"Condition folder: {self.cond_dir}")
        print(f"JSONs folder: {self.jsons_dir}")

        if len(self.valid_samples) == 0:
            raise ValueError(f"No valid GT-Condition-JSON triplets found in {data_dir}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """
        返回与原始webdataset完全兼容的格式，但内部使用预先准备的条件图像
        {
            'jpg': PIL.Image (GT图像),
            'condition_img': PIL.Image (预先准备的条件图像),
            'json': {'prompt': 'xxx'},
            '__key__': 'filename',
            '__url__': 'path'
        }
        """
        sample_info = self.valid_samples[idx]

        try:
            # 1. 加载GT图像
            gt_image = Image.open(sample_info["gt_img_path"]).convert("RGB")
            if gt_image.size != self.target_size:
                gt_image = gt_image.resize(self.target_size, Image.LANCZOS)

            # 2. 加载预先准备的条件图像
            cond_image = Image.open(sample_info["cond_img_path"]).convert("RGB")
            if cond_image.size != self.condition_size:
                cond_image = cond_image.resize(self.condition_size, Image.LANCZOS)

            # 3. 加载JSON数据
            with open(sample_info["json_path"], "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # 确保json格式正确
            if isinstance(json_data, str):
                json_data = json.loads(json_data)

            # 确保有prompt字段
            if "prompt" not in json_data:
                if "text" in json_data:
                    json_data["prompt"] = json_data["text"]
                elif "caption" in json_data:
                    json_data["prompt"] = json_data["caption"]
                elif "description" in json_data:
                    json_data["prompt"] = json_data["description"]
                else:
                    json_data["prompt"] = ""
                    print(f"Warning: No prompt found in {sample_info['json_path']}, using empty string")

            # 4. 返回数据（兼容webdataset格式，但增加了预先准备的条件图像）
            return {
                "jpg": gt_image,  # GT图像（用于训练目标）
                "condition_img": cond_image,  # 预先准备的条件图像
                "json": json_data,
                "__key__": sample_info["key"],
                "__url__": sample_info["url"],
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            print(f"GT: {sample_info['gt_img_path']}")
            print(f"Cond: {sample_info['cond_img_path']}")
            print(f"JSON: {sample_info['json_path']}")
            
            # 返回空白样本
            blank_gt = Image.new("RGB", self.target_size, (0, 0, 0))
            blank_cond = Image.new("RGB", self.condition_size, (0, 0, 0))
            return {
                "jpg": blank_gt,
                "condition_img": blank_cond,
                "json": {"prompt": ""},
                "__key__": f"error_{idx}",
                "__url__": sample_info["url"],
            }

def create_custom_dataset_wrapper(base_dataset, **kwargs):
    """
    包装自定义数据集，使其可以与原有的ImageConditionDataset无缝结合
    """
    return ImageConditionDataset(base_dataset, **kwargs)


@torch.no_grad()
def test_function(model, save_path, file_name):

    # NOTE: add condition_img path
    cond_test_imgpath = model.training_config["dataset"]["cond_test_imgpath"]
    cond_test_imgprompt = model.training_config["dataset"]["cond_test_imgprompt"]

    condition_size = model.training_config["dataset"]["condition_size"]
    target_size = model.training_config["dataset"]["target_size"]

    position_delta = model.training_config["dataset"].get("position_delta", [0, 0])
    position_scale = model.training_config["dataset"].get("position_scale", 1.0)

    adapter = model.adapter_names[2]
    condition_type = model.training_config["condition_type"]
    test_list = []

    if condition_type in ["canny", "coloring", "deblurring", "depth"]:
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition_img = convert_to_condition(condition_type, image, 5)
        condition = Condition(condition_img, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "depth_pred":
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition = Condition(image, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "fill":
        condition_img = (
            Image.open("./assets/vase_hq.jpg").resize(condition_size).convert("RGB")
        )
        mask = Image.new("L", condition_img.size, 0)
        draw = ImageDraw.Draw(mask)
        a = condition_img.size[0] // 4
        b = a * 3
        draw.rectangle([a, a, b, b], fill=255)
        condition_img = Image.composite(
            condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
        )
        condition = Condition(condition, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "super_resolution":
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition = Condition(image, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "lineart" or condition_type == "sketch":
        # TODO: 固定一组条件图像专门用于训练时验证
        image = Image.open(cond_test_imgpath)
        image = image.resize(condition_size)
        condition = Condition(image, adapter, position_delta, position_scale)
        test_list.append((condition, cond_test_imgprompt))
        
    else:
        raise NotImplementedError
    os.makedirs(save_path, exist_ok=True)
    for i, (condition, prompt) in enumerate(test_list):
        generator = torch.Generator(device=model.device)
        generator.manual_seed(42)

        res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
        )
        # NOTE: 加上了条件图和生成图一起显示

        # file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
        # res.images[0].save(file_path)

        # 生成结果
        gen_img = res.images[0].convert("RGB").resize((target_size[0], target_size[1]))

        # 条件图（直接取 Condition 里保存的）
        cond_img = condition.condition.convert("RGB").resize((target_size[0], target_size[1]))

        # 横向拼接：左边条件图，右边生成图
        concat_img = Image.new("RGB", (target_size[0] * 2, target_size[1]))
        concat_img.paste(cond_img, (0, 0))
        concat_img.paste(gen_img, (target_size[0], 0))

        # 保存
        file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
        concat_img.save(file_path)



def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # 检查是否使用自定义数据集
    if training_config["dataset"].get("custom", False) == True:
        # 使用自定义数据集
        print("Using custom dataset...")
        base_dataset = CustomImageConditionDataset(
            data_dir=training_config["dataset"].get("data_dir", None)
        )
    else:
        # 使用原有的webdataset， 即Load dataset text-to-image-2M
        print("Using webdataset...")
        base_dataset = load_dataset(
            "webdataset",
            data_files={"train": training_config["dataset"]["urls"]},
            split="train",
            cache_dir="cache/t2i2m",
            num_proc=32,
        )

    # Initialize custom dataset (统一使用ImageConditionDataset包装)
    dataset = ImageConditionDataset(
            base_dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
            position_scale=training_config["dataset"].get("position_scale", 1.0),
        )

    # import ipdb;ipdb.set_trace();

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
