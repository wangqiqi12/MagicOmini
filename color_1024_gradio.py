import os
import torch
import gradio as gr
import numpy as np
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageDraw
import cv2
import argparse

from omini.pipeline.flux_omini import Condition, generate, seed_everything

# 全局变量
N_POINTS = 1
USE_CPU = False  # 是否使用CPU运行
DEVICE = "cuda"  # 默认设备

pipe = None
edit_confirmed = False  # 编辑确认状态
color_confirmed = False  # 颜色提示确认状态
current_edit_data = None  # 当前编辑数据
current_color_data = None  # 当前颜色提示数据
last_sketch_hash = None  # 上次编辑数据的哈希值，用于检测变化
last_color_hash = None  # 上次颜色数据的哈希值
refresh_counter = 0  # 强制刷新计数器
local_backup = {}  # 本地状态备份，防止网络中断丢失状态
click_timestamps = []  # 点击时间戳，用于检测网络延迟
status_check_counter = 0  # 状态检查计数器
last_operation_time = 0  # 最后操作时间

def initialize_pipeline():
    """初始化pipeline - 支持CPU/GPU选择，返回状态信息"""
    global pipe, USE_CPU, DEVICE
    
    if pipe is not None:
        return True, f"✅ Pipeline已就绪 (设备: {DEVICE})"
    
    print("🔄 正在初始化pipeline...")
    
    try:
        # 检查设备可用性
        if USE_CPU:
            DEVICE = "cpu"
            print("⚙️ 使用CPU模式")
        else:
            if not torch.cuda.is_available():
                print("⚠️ CUDA不可用，自动切换到CPU模式")
                DEVICE = "cpu"
                USE_CPU = True
            else:
                DEVICE = "cuda"
                print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
        
        # NOTE: 请修改为你的实际模型路径
        local_path = "/root/private_data/wangqiqi12/Omini_ckpts/FLUX.1-dev"
        
        print(f"📂 加载基础模型: {local_path}")
        # 根据设备选择加载模型
        if USE_CPU:
            print("⚙️ 使用CPU加载模型（可能较慢）...")
            pipe = FluxPipeline.from_pretrained(
                local_path,
                torch_dtype=torch.float32,  # CPU使用float32
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        else:
            pipe = FluxPipeline.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,  # GPU使用bfloat16
                device_map="cuda",
                low_cpu_mem_usage=True,
            )
        
        # 验证模型设备信息
        try:
            if hasattr(pipe, 'device'):
                print(f"🔍 Pipeline设备: {pipe.device}")
            elif hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'device'):
                print(f"🔍 Transformer设备: {pipe.transformer.device}")
            print(f"🎯 当前运行设备: {DEVICE}")
        except:
            pass
        
        print("📦 加载LoRA权重...")
        # NOTE: 请修改为你的实际LoRA路径  
        lora_path = "/root/private_data/wangqiqi12/Omini_ckpts/omni_ckpts/color_sketch_1024"
        
        # 检查LoRA文件是否存在
        import os
        lora_file = os.path.join(lora_path, "default.safetensors")
        if not os.path.exists(lora_file):
            error_msg = f"❌ LoRA文件不存在: {lora_file}"
            print(error_msg)
            print("   请检查路径或权重文件名")
            return False, error_msg
        
        pipe.load_lora_weights(
            lora_path,
            weight_name="default.safetensors",
            adapter_name="sketch",
        )
        
        # 设置为评估模式并优化内存
        pipe.unet.eval() if hasattr(pipe, 'unet') else None
        pipe.transformer.eval() if hasattr(pipe, 'transformer') else None
        
        # 启用内存优化（与device_map兼容）
        try:
            pipe.enable_attention_slicing()
            print("✅ 已启用注意力切片以节省内存")
        except Exception as e:
            print(f"⚠️ 注意力切片不可用: {e}")
        
        # 设备特定优化
        if USE_CPU:
            print("💡 CPU模式：推荐使用较少的推理步数以加快速度")
        else:
            print("💡 使用device_map自动管理GPU内存")
        
        print(f"🎉 Pipeline初始化完成! (设备: {DEVICE})")
        return True, f"✅ Pipeline初始化完成! (设备: {DEVICE})"
        
    except FileNotFoundError as e:
        error_msg = f"❌ 文件未找到: {str(e)}"
        print(error_msg)
        return False, error_msg
    except torch.cuda.OutOfMemoryError as e:
        error_msg = f"❌ GPU内存不足: {str(e)}"
        print(error_msg)
        print("💡 建议: 尝试重启程序或使用更小的模型")
        return False, error_msg
    except Exception as e:
        error_msg = f"❌ Pipeline初始化失败: {str(e)}"
        print(error_msg)
        return False, error_msg

def create_masked_image_from_sketch(base_image, sketch_data):
    """从base图像和用户编辑的sketch创建masked图像"""
    if base_image is None:
        return None, "请先上传基础图像"
    
    if sketch_data is None:
        return None, "请先在图像上编辑mask区域"
    
    try:
        # 处理base图像
        if isinstance(base_image, np.ndarray):
            base_image = Image.fromarray(base_image.astype(np.uint8))
        base_image = base_image.resize((1024, 1024)).convert('RGB')
        
        # 处理用户编辑后的图像数据
        edited_image = None
        
        print(f"Debug: sketch_data type: {type(sketch_data)}")
        
        if isinstance(sketch_data, dict):
            print(f"Debug: sketch_data keys: {sketch_data.keys()}")
            # 尝试常见的字段名 - 优先使用composite（合成图像）
            possible_keys = ['composite', 'image', 'background', 'layers', 'data']
            for key in possible_keys:
                if key in sketch_data and sketch_data[key] is not None:
                    edited_image = sketch_data[key]
                    print(f"Debug: 使用字段 '{key}'")
                    break
        elif isinstance(sketch_data, np.ndarray):
            edited_image = sketch_data
        elif isinstance(sketch_data, Image.Image):
            edited_image = sketch_data
        else:
            edited_image = sketch_data
        
        if edited_image is None:
            return None, "无法从编辑数据中提取图像"
        
        # 统一转换为PIL图像    
        if isinstance(edited_image, np.ndarray):
            # 确保数据类型正确
            if edited_image.dtype != np.uint8:
                edited_image = (edited_image * 255).astype(np.uint8) if edited_image.max() <= 1 else edited_image.astype(np.uint8)
            edited_image = Image.fromarray(edited_image)
        elif not isinstance(edited_image, Image.Image):
            return None, f"不支持的图像数据类型: {type(edited_image)}"
        
        # 调整尺寸和格式
        edited_image = edited_image.resize((1024, 1024)).convert('RGB')
        
        # 创建mask：检测白色涂抹区域
        edited_array = np.array(edited_image)
        base_array = np.array(base_image)
        
        # 找到用户用白色笔涂抹的区域（接近白色的像素）
        # 白色涂抹区域：RGB值都很高（接近255）
        white_mask = ((edited_array[:,:,0] > 240) & 
                     (edited_array[:,:,1] > 240) & 
                     (edited_array[:,:,2] > 240))
        
        # 同时排除原图中本来就是白色的区域
        original_white = ((base_array[:,:,0] > 240) & 
                         (base_array[:,:,1] > 240) & 
                         (base_array[:,:,2] > 240))
        
        # 真正的mask区域：编辑后是白色但原图不是白色的区域
        mask_region = white_mask & ~original_white
        
        # 如果没有检测到明显的白色涂抹，则检测所有明显变化的区域
        if np.sum(mask_region) < 100:  # 如果mask区域太小
            # 计算像素差异
            diff = np.abs(edited_array.astype(np.float32) - base_array.astype(np.float32))
            diff_magnitude = np.sum(diff, axis=2)
            # 找到差异较大的区域（用户编辑过的区域）
            mask_region = diff_magnitude > 30
        
        # 创建最终的masked图像
        # 用户编辑后的图像就是我们要的条件图像
        masked_image = edited_image.copy()
        
        # 统计mask区域
        mask_pixels = np.sum(mask_region)
        total_pixels = mask_region.size
        mask_percentage = (mask_pixels / total_pixels) * 100
        
        status = f"编辑检测成功! Mask区域: {mask_pixels} 像素 ({mask_percentage:.1f}%)"
        
        return masked_image, status
        
    except Exception as e:
        return None, f"处理编辑图像失败: {str(e)}"

def extract_color_hints_from_strokes(stroke_image, original_cond_image, radius=5, n_points=70):
    """从颜色笔触中直接提取纯色方块 - 参考color_hint_ui.py"""
    if stroke_image is None or original_cond_image is None:
        return None
    
    # 确保stroke_image是numpy数组
    if isinstance(stroke_image, Image.Image):
        stroke_array = np.array(stroke_image)
    else:
        stroke_array = stroke_image
        
    h, w = stroke_array.shape[:2]
    
    # 检测原始条件图中的白色mask区域
    if isinstance(original_cond_image, Image.Image):
        original_cond_array = np.array(original_cond_image)
    else:
        original_cond_array = original_cond_image
        
    original_gray = cv2.cvtColor(original_cond_array, cv2.COLOR_RGB2GRAY)
    white_mask_area = original_gray > 240  # 白色区域

    # 如果条件图中没有明显的白色mask（用户可能没有严格使用白色遮罩），
    # 不要直接放弃；改为将整个图像作为候选区域以便提取颜色提示（更宽松的容错处理）。
    if not np.any(white_mask_area):
        # 宽松回退：允许全图作为mask区域，但会在后续步骤中仍然检测颜色和差异
        white_mask_area = np.ones_like(original_gray, dtype=bool)
    
    # 检查图像形状是否匹配
    if stroke_array.shape != original_cond_array.shape:
        return original_cond_image
    
    # 计算编辑前后的差异，找到新添加的颜色stroke
    diff = np.abs(stroke_array.astype(np.float32) - original_cond_array.astype(np.float32))
    diff_sum = np.sum(diff, axis=2)
    
    # 检测有明显变化的区域
    significant_change = diff_sum > 30
    
    # 检测stroke_image中的颜色（排除极端黑/白），阈值稍微放宽以捕捉更淡的颜色笔触
    stroke_gray = cv2.cvtColor(stroke_array, cv2.COLOR_RGB2GRAY)
    has_color = (stroke_gray > 20) & (stroke_gray < 245)  # 不是极暗也不是几乎纯白
    
    # 找到既在白色mask区域、又有颜色、又是新添加的像素
    valid_color_indices = np.argwhere(significant_change & has_color & white_mask_area)
    
    if len(valid_color_indices) == 0:
        return original_cond_image
    
    # 创建新的条件图，从原图开始
    new_cond_array = original_cond_array.copy()
    
    # 从有颜色的像素中随机采样
    n_sample = min(n_points, len(valid_color_indices))
    sampled_indices = valid_color_indices[np.random.choice(
        len(valid_color_indices), size=n_sample, replace=False)]
    
    n_valid = 0
    for y, x in sampled_indices:
        # 边界检查
        if y - radius < 0 or y + radius >= h or x - radius < 0 or x + radius >= w:
            continue
        
        # 检查这个patch是否完全在白色mask内
        patch_white_area = white_mask_area[y - radius:y + radius + 1, x - radius:x + radius + 1]
        if patch_white_area.shape != (2 * radius + 1, 2 * radius + 1):
            continue
        if not np.all(patch_white_area):
            continue  # 不完全在白色区域内，跳过
        
        # 获取颜色并设置color hint
        raw_color = stroke_array[y, x]
        
        # 确保颜色值不为白色（255,255,255）
        if np.all(raw_color >= 248):
            # 如果颜色太接近白色，稍微调暗
            color = np.clip(raw_color.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
        else:
            color = raw_color
            
        # 重要：创建一个固定纯色值，确保整个方块都是完全相同的颜色
        fixed_color = [int(color[0]), int(color[1]), int(color[2])]
        
        # 直接填充整个方块为这个纯色
        new_cond_array[y - radius:y + radius + 1, x - radius:x + radius + 1] = fixed_color
        
        n_valid += 1
        
        if n_valid >= n_points:
            break  # 达到期望数量就停止
    
    # 返回PIL图像
    if isinstance(original_cond_image, Image.Image):
        return Image.fromarray(new_cond_array.astype(np.uint8))
    else:
        return new_cond_array

def create_color_condition_image(base_image, sketch_data, color_stroke_data):
    """创建带颜色提示的条件图像"""
    try:
        if base_image is None or sketch_data is None:
            return None, "请先完成sketch编辑"
        
        # 首先创建基础的masked图像（包含sketch）
        masked_image, status = create_masked_image_from_sketch(base_image, sketch_data)
        if masked_image is None:
            return None, f"创建基础条件图失败: {status}"
        
        # 如果没有颜色笔触数据，直接返回基础条件图
        if color_stroke_data is None:
            return masked_image, "使用基础条件图（无颜色提示）"
        
        # 处理颜色笔触数据
        color_image = None
        if isinstance(color_stroke_data, dict) and 'composite' in color_stroke_data:
            color_image = color_stroke_data['composite']
        elif isinstance(color_stroke_data, np.ndarray):
            color_image = Image.fromarray(color_stroke_data.astype(np.uint8))
        elif isinstance(color_stroke_data, Image.Image):
            color_image = color_stroke_data
        
        if color_image is None:
            return masked_image, "颜色数据无效，使用基础条件图"
        
        # 确保图像格式一致
        if isinstance(color_image, np.ndarray):
            color_image = Image.fromarray(color_image.astype(np.uint8))
        color_image = color_image.resize((1024, 1024)).convert('RGB')
        
        # 提取颜色提示并生成最终条件图
        final_cond_image = extract_color_hints_from_strokes(color_image, masked_image, radius=5, n_points=N_POINTS)
        
        if final_cond_image is None:
            return masked_image, "颜色提示提取失败，使用基础条件图"
        
        return final_cond_image, "✅ 条件图已生成（包含颜色提示）"
        
    except Exception as e:
        return None, f"创建颜色条件图失败: {str(e)}"

def save_condition_image(cond_image):
    """保存条件图像为PNG格式"""
    if cond_image is None:
        return None, "没有条件图可以保存"
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("gradio_output", exist_ok=True)
    
    try:
        filename = f"gradio_output/condition_{timestamp}.png"
        if isinstance(cond_image, np.ndarray):
            cond_pil = Image.fromarray(cond_image.astype(np.uint8))
        else:
            cond_pil = cond_image
        
        cond_pil.save(filename, format='PNG', optimize=True)
        return filename, f"✅ 条件图已保存为: {filename}"
    except Exception as e:
        return None, f"❌ 保存失败: {str(e)}"

def generate_image(prompt, num_steps, guidance_scale):
    """生成图像的主函数 - 使用确认的颜色条件图"""
    global edit_confirmed, color_confirmed, current_edit_data, current_color_data
    
    try:
        print(f"🔄 开始生成图像...")
        
        # 检查编辑确认状态
        if not edit_confirmed or current_edit_data is None:
            return None, None, "❌ 请先点击'确认编辑完成'按钮"
        
        # 检查是否有颜色条件图
        if not color_confirmed or current_color_data is None:
            return None, None, "❌ 请先点击'确认颜色提示'按钮"
        
        base_image = current_edit_data['base_image']
        sketch_data = current_edit_data['sketch_data']
        
        if not prompt.strip():
            return None, None, "❌ 请输入prompt描述"
        
        # 初始化pipeline - 强制CUDA检查
        print("⏳ 正在初始化模型...")
        try:
            success, init_msg = initialize_pipeline()
            if not success:
                return None, None, init_msg
            print(init_msg)
        except Exception as init_error:
            print(f"Pipeline初始化错误: {init_error}")
            return None, None, f"❌ 模型初始化失败: {str(init_error)}"
        
        print("🖼️ 使用内存中的条件图（无需PNG文件）...")
        try:
            if current_color_data is None or 'condition_image' not in current_color_data:
                return None, None, "❌ 条件图缺失，请重新生成颜色提示"

            masked_image = current_color_data['condition_image']
            # 确保为PIL图像
            if isinstance(masked_image, np.ndarray):
                masked_image = Image.fromarray(masked_image.astype(np.uint8))
            masked_image = masked_image.convert('RGB').resize((1024, 1024))
            print(f"✅ 已加载内存条件图，尺寸: {masked_image.size}")

        except Exception as mask_error:
            print(f"条件图加载错误: {mask_error}")
            return None, None, f"❌ 条件图加载失败: {str(mask_error)}"
        
        print("🎯 准备生成条件...")
        # 创建condition - 添加安全检查
        try:
            # 确保LoRA适配器正确加载
            if hasattr(pipe, 'set_adapters'):
                pipe.set_adapters("sketch")
            condition = Condition(masked_image, "sketch")
        except Exception as condition_error:
            print(f"条件准备错误: {condition_error}")
            return None, None, f"❌ 条件准备失败: {str(condition_error)}"
        
        # 设置随机种子
        seed_everything(42)
        
        # 生成图像 - 添加进度提示
        print(f"🚀 正在生成图像 (步数: {int(num_steps)}, 引导强度: {guidance_scale})")
        print(f"📝 Prompt: {prompt}")
        
        try:
            # 使用torch.no_grad()优化内存使用
            with torch.no_grad():
                result = generate(
                    pipe,
                    prompt=prompt,
                    conditions=[condition],
                    height=1024,
                    width=1024,
                    num_inference_steps=int(num_steps),
                    guidance_scale=guidance_scale,
                )
            
            if result is None or len(result.images) == 0:
                return None, None, "❌ 生成失败：模型返回空结果"
                
            result_img = result.images[0]
            print("✅ 图像生成完成")
            
        except Exception as gen_error:
            print(f"生成过程错误: {gen_error}")
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None, f"❌ 生成失败: {str(gen_error)}"
        
        print("🔗 正在创建对比图...")
        # 创建对比图像 - 优化内存使用
        try:
            concat_image = Image.new("RGB", (1024 * 3, 1024))
            base_resized = base_image.resize((1024, 1024)).convert('RGB') if base_image else Image.new("RGB", (1024, 1024), (255, 255, 255))
            concat_image.paste(base_resized, (0, 0))
            # 中间展示：使用生成颜色提示块前的原始带颜色笔触图（来自current_color_data['color_stroke_data']）
            stroke_img = None
            try:
                if current_color_data and 'color_stroke_data' in current_color_data and current_color_data['color_stroke_data'] is not None:
                    stroke_raw = current_color_data['color_stroke_data']
                    if isinstance(stroke_raw, dict) and 'composite' in stroke_raw:
                        stroke_img = stroke_raw['composite']
                    elif isinstance(stroke_raw, np.ndarray):
                        stroke_img = Image.fromarray(stroke_raw.astype(np.uint8))
                    elif isinstance(stroke_raw, Image.Image):
                        stroke_img = stroke_raw
                # 如果没有raw stroke图像，回退到mask图
                if stroke_img is None:
                    stroke_img = masked_image
                if isinstance(stroke_img, np.ndarray):
                    stroke_img = Image.fromarray(stroke_img.astype(np.uint8))
                stroke_img = stroke_img.resize((1024, 1024)).convert('RGB')
            except Exception:
                stroke_img = masked_image

            concat_image.paste(stroke_img, (1024, 0))
            concat_image.paste(result_img, (1024 * 2, 0))
        except Exception as concat_error:
            print(f"对比图创建错误: {concat_error}")
            # 即使对比图失败，也返回生成结果
            return result_img, None, "⚠️ 生成成功，但对比图创建失败"
        
        # 优化保存逻辑 - 异步保存，避免阻塞Gradio状态
        try:
            import threading
            
            def save_images_async():
                try:
                    os.makedirs("gradio_output", exist_ok=True)
                    result_img.save("gradio_output/result.jpg", quality=95, optimize=True)
                    concat_image.save("gradio_output/comparison.jpg", quality=95, optimize=True)
                    print("💾 结果已异步保存到 gradio_output/")
                except Exception as save_error:
                    print(f"异步保存错误: {save_error}")
            
            # 启动后台保存线程，不阻塞主流程
            save_thread = threading.Thread(target=save_images_async, daemon=True)
            save_thread.start()
            
        except Exception as save_error:
            print(f"保存线程启动失败: {save_error}")
            # 保存失败不影响返回结果
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result_img, concat_image, "🎉 生成成功！"
        
    except Exception as e:
        error_msg = f"❌ 生成失败: {str(e)}"
        print(error_msg)
        # 发生错误时清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None, error_msg

def update_sketch_pad(base_image):
    """当上传新图像时，更新绘制画布的背景并重置编辑状态"""
    global edit_confirmed, current_edit_data, last_sketch_hash
    
    # 重置编辑状态
    edit_confirmed = False
    current_edit_data = None
    last_sketch_hash = None
    
    if base_image is None:
        return np.ones((1024, 1024, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 将PIL图像转换为numpy数组
    if isinstance(base_image, Image.Image):
        # 调整到1024 * 1024并转换为numpy数组
        resized_image = base_image.resize((1024, 1024)).convert('RGB')
        return np.array(resized_image)
    elif isinstance(base_image, np.ndarray):
        return base_image
    else:
        return np.ones((1024, 1024, 3), dtype=np.uint8) * 255

def check_sketch_changes(sketch_data):
    """检测编辑区域是否有变化，并重置确认状态"""
    global edit_confirmed, last_sketch_hash
    import hashlib
    import time
    
    if sketch_data is None:
        return "⏳ 等待编辑..."
    
    try:
        # 计算当前编辑数据的哈希值
        if isinstance(sketch_data, dict) and 'composite' in sketch_data:
            data_to_hash = sketch_data['composite']
        else:
            data_to_hash = sketch_data
            
        if isinstance(data_to_hash, np.ndarray):
            current_hash = hashlib.md5(data_to_hash.tobytes()).hexdigest()
        else:
            current_hash = str(hash(str(data_to_hash)))
        
        # 检查是否有变化
        if last_sketch_hash != current_hash:
            if last_sketch_hash is not None:  # 不是第一次
                edit_confirmed = False  # 重置确认状态
                timestamp = time.strftime("%H:%M:%S")
                print(f"🔄 [{timestamp}] 检测到编辑变化，重置确认状态")
                return "🔄 检测到编辑变化，请重新确认编辑"
            last_sketch_hash = current_hash
            
        if edit_confirmed:
            return "✅ 编辑已确认"
        else:
            return "⏳ 请点击确认编辑按钮"
            
    except Exception as e:
        return f"⚠️ 状态检查错误: {str(e)}"

def get_current_status():
    """获取当前完整状态 - 独立查询函数，不依赖网络同步"""
    global edit_confirmed, color_confirmed, current_edit_data, current_color_data, status_check_counter
    import time
    
    status_check_counter += 1
    timestamp = time.strftime("%H:%M:%S")
    
    # 状态检查逻辑
    if edit_confirmed and color_confirmed and current_edit_data and current_color_data:
        main_status = f"✅ [{timestamp}] 全部确认完成 - 检查#{status_check_counter}"
        edit_status = "✅ 编辑已确认"
        color_status = "✅ 颜色已确认，可以生成图像"
        network_status = "🟢 状态同步正常"
    elif edit_confirmed and current_edit_data:
        main_status = f"🟡 [{timestamp}] 编辑已确认，等待颜色提示 - 检查#{status_check_counter}"
        edit_status = "✅ 编辑已确认"
        color_status = "⏳ 请添加颜色提示并确认"
        network_status = "� 等待颜色确认"
    else:
        main_status = f"⏳ [{timestamp}] 等待编辑确认 - 检查#{status_check_counter}"
        edit_status = "⏳ 请完成编辑并确认"
        color_status = "⏳ 等待编辑完成"
        network_status = f"🟡 未确认状态 - 检查#{status_check_counter}"
    
    print(f"📊 [{timestamp}] 状态查询#{status_check_counter}: edit={edit_confirmed}, color={color_confirmed}")
    return main_status, edit_status, color_status, network_status

def backup_state():
    """备份当前状态到本地"""
    global local_backup, edit_confirmed, current_edit_data, last_sketch_hash, last_operation_time
    import time
    
    last_operation_time = time.time()
    local_backup = {
        'edit_confirmed': edit_confirmed,
        'current_edit_data': current_edit_data,
        'last_sketch_hash': last_sketch_hash,
        'timestamp': last_operation_time
    }
    print(f"💾 状态已备份: confirmed={edit_confirmed}")

def restore_state():
    """从本地备份恢复状态"""
    global local_backup, edit_confirmed, current_edit_data, last_sketch_hash
    import time
    
    if local_backup and time.time() - local_backup.get('timestamp', 0) < 300:  # 5分钟内的备份有效
        edit_confirmed = local_backup.get('edit_confirmed', False)
        current_edit_data = local_backup.get('current_edit_data')
        last_sketch_hash = local_backup.get('last_sketch_hash')
        print(f"📥 状态已恢复: confirmed={edit_confirmed}")
        return True
    return False

def continue_editing(base_image):
    """重新编辑 - 完全重置所有状态"""
    global edit_confirmed, color_confirmed, current_edit_data, current_color_data, last_sketch_hash, last_color_hash, refresh_counter
    import time
    
    timestamp = time.strftime("%H:%M:%S")
    print(f"🔄 [{timestamp}] 重新编辑 - 重置所有状态")
    
    # 完全重置所有状态
    edit_confirmed = False
    color_confirmed = False
    current_edit_data = None
    current_color_data = None
    last_sketch_hash = None
    last_color_hash = None
    refresh_counter += 1
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 准备UI更新数据
    main_status = f"🔄 [{timestamp}] 已重置 (刷新#{refresh_counter})"
    edit_status = "⏳ 请完成编辑并确认"
    color_status = "⏳ 等待编辑完成"
    network_status = "🟢 状态已重置"
    
    # 恢复原图
    sketch_pad_result = update_sketch_pad(base_image)
    color_pad_result = update_sketch_pad(base_image)
    
    print(f"✅ [{timestamp}] 重置完成 (刷新#{refresh_counter})")
    
    return sketch_pad_result, main_status, edit_status, color_status, network_status, color_pad_result, None, ""

def confirm_edit_ready(base_image, sketch_data):
    """确认编辑就绪 - 简化版本"""
    global edit_confirmed, current_edit_data, last_sketch_hash, color_confirmed, current_color_data
    
    import time
    import hashlib
    from datetime import datetime
    timestamp = time.strftime("%H:%M:%S")
    
    print(f"✅ [{timestamp}] 开始确认编辑")
    
    # 重置状态
    edit_confirmed = False
    current_edit_data = None
    color_confirmed = False
    current_color_data = None
    
    if base_image is None:
        return "❌ 请先上传基础图像", None
    
    if sketch_data is None:
        return "❌ 请先在图像上进行编辑", None
    
    try:
        # 处理编辑数据
        masked_image, status = create_masked_image_from_sketch(base_image, sketch_data)
        
        if masked_image is None:
            return f"❌ 编辑数据无效: {status}", None
        
        # 保存基础条件图为PNG
        os.makedirs("condition_images", exist_ok=True)
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_condition_png_path = f"condition_images/base_condition_{file_timestamp}.png"
        
        if isinstance(masked_image, np.ndarray):
            masked_pil = Image.fromarray(masked_image.astype(np.uint8))
        else:
            masked_pil = masked_image
        
        masked_pil.save(base_condition_png_path, format='PNG', optimize=True)
        
        # 更新状态
        edit_confirmed = True
        current_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        last_sketch_hash = current_hash
        
        current_edit_data = {
            'base_image': base_image,
            'sketch_data': sketch_data,
            'masked_image': masked_image,
            'base_condition_png_path': base_condition_png_path,
            'hash': current_hash,
            'timestamp': timestamp
        }
        
        success_msg = f"✅ [{timestamp}] 编辑已确认！{status}"
        print(f"✅ [{timestamp}] 确认完成")
        
        return success_msg, masked_image
        
    except Exception as e:
        edit_confirmed = False
        error_msg = f"❌ [{timestamp}] 确认失败: {str(e)}"
        print(error_msg)
        return error_msg, None

def generate_color_hints_from_strokes(color_stroke_data):
    """从颜色笔触生成颜色提示块"""
    global edit_confirmed, current_edit_data, color_confirmed, current_color_data
    
    import time
    from datetime import datetime
    timestamp = time.strftime("%H:%M:%S")
    
    print(f"🎨 [{timestamp}] 生成颜色提示块")
    
    if not edit_confirmed or current_edit_data is None:
        return "❌ 请先确认编辑完成", None
    
    try:
        base_image = current_edit_data['base_image']
        sketch_data = current_edit_data['sketch_data']
        masked_image = current_edit_data['masked_image']
        
        if color_stroke_data is None:
            condition_image = masked_image
            status_msg = "⚠️ 没有颜色笔触，使用基础条件图"
        else:
            condition_image, status = create_color_condition_image(base_image, sketch_data, color_stroke_data)

            if condition_image is None:
                return f"❌ 颜色条件图创建失败: {status}", None

            status_msg = f"✅ 颜色提示块已生成！{status}"

        # 不再保存为PNG文件，直接使用内存中的条件图
        # 更新颜色条件数据并自动确认
        color_confirmed = True
        current_color_data = {
            'condition_image': condition_image,
            'color_stroke_data': color_stroke_data,
            'timestamp': timestamp,
            'confirmed': True
        }

        final_msg = f"{status_msg} (已自动确认，并使用内存条件图)"
        print(f"✅ [{timestamp}] 颜色提示块生成并自动确认完成")

        return final_msg, condition_image
        
    except Exception as e:
        error_msg = f"❌ [{timestamp}] 生成失败: {str(e)}"
        print(error_msg)
        return error_msg, None

def confirm_color_hints_ready():
    """确认颜色提示准备就绪"""
    global color_confirmed, current_color_data
    
    import time
    timestamp = time.strftime("%H:%M:%S")
    
    print(f"✅ [{timestamp}] 确认颜色提示")
    
    if current_color_data is None:
        return "❌ 请先点击'生成颜色提示块'按钮", None
    
    try:
        color_confirmed = True
        current_color_data['confirmed'] = True
        
        condition_image = current_color_data['condition_image']
        
        success_msg = f"✅ [{timestamp}] 颜色提示已确认！可以生成图像"
        print(f"✅ [{timestamp}] 颜色确认完成")
        
        return success_msg, condition_image
        
    except Exception as e:
        color_confirmed = False
        error_msg = f"❌ [{timestamp}] 确认失败: {str(e)}"
        print(error_msg)
        return error_msg, None

# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="OminiControl Inpainting Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 OminiControl Inpainting Demo")
        gr.Markdown("**使用说明**: 上传图像 → 编辑mask/sketch并确认 → 添加颜色并生成颜色块 → 确认颜色 → 生成图像 | **响应慢?** 点击📊查询状态")
        
        # 横向布局：上传图像、sketch编辑区、color编辑区
        with gr.Row():
            with gr.Column(scale=1):
                base_image = gr.Image(
                    label="📤 1. 上传基础图像",
                    type="pil",
                    height=768,
                    width=768
                )
            with gr.Column(scale=1):
                sketch_pad = gr.ImageEditor(
                    label="🖌️ 2. 在原图上编辑 (白笔涂抹mask区域，黑笔勾勒sketch)",
                    type="numpy",
                    height=768,
                    brush=gr.Brush(
                        default_size=15,
                        colors=["#FFFFFF", "#000000"],
                        default_color="#FFFFFF"
                    ),
                    value=np.ones((1024, 1024, 3), dtype=np.uint8) * 255
                )
            with gr.Column(scale=1):
                color_pad = gr.ImageEditor(
                    label="🎨 4. 添加颜色提示 (在mask区域画颜色笔触)",
                    type="numpy",
                    height=768,
                    brush=gr.Brush(
                        default_size=10,
                        colors=["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#800080"],
                        default_color="#FF0000"
                    ),
                    value=np.ones((1024, 1024, 3), dtype=np.uint8) * 255
                )
            
            # final condition image display removed (kept in memory only)


        # 专门的一栏：展示生成的带颜色提示的条件图（保存在内存中）
        with gr.Row():
            condition_preview = gr.Image(
                label="🔍 生成的条件图（含颜色提示）",
                type="pil",
                height=512,
                width=512
            )


        # 控制按钮区域 
        with gr.Row():
            clear_btn = gr.Button("🗑️ 重置为原图", variant="secondary", size="sm")
            confirm_btn = gr.Button("✅ 确认编辑", variant="primary", size="sm")
            # generate_color_btn moved to dedicated section with n_points slider
        
        # 颜色提示控制区域
        gr.Markdown("### 🎨 颜色提示设置")
        with gr.Row():
            with gr.Column(scale=2):
                n_points_slider = gr.Slider(
                                minimum=1,
                                maximum=70,
                                value=N_POINTS,
                                step=1,
                                label="🎯 颜色提示块数量",
                                info="控制从颜色笔触中提取多少个颜色方块（1-70个）"
                            )
            with gr.Column(scale=1):
                confirm_generate_color_btn = gr.Button(
                    "🎨 确认生成颜色提示块", 
                    variant="primary", 
                    size="lg"
                )
        
        # 参数控制区域
        gr.Markdown("### ⚙️ 生成参数")
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="✏️ 输入Prompt描述",
                    placeholder="描述你想要在mask区域生成的内容，例如：A beautiful flower vase",
                    lines=2,
                    value="A beautiful vase"
                )
            with gr.Column(scale=1):
                num_steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=28,
                    step=1,
                    label="推理步数"
                )
            with gr.Column(scale=1):
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.5,
                    step=0.1,
                    label="引导强度"
                )
        
        # 生成按钮
        with gr.Row():
            generate_btn = gr.Button("🚀 生成图像", variant="primary", size="lg")
        
        # 状态显示
        with gr.Row():
            status_text = gr.Textbox(
                label="📝 状态信息",
                value="请上传图像并绘制mask区域",
                interactive=False,
                lines=1
            )
        
        # （已简化）主状态使用上面的 `status_text`
        
        # 结果显示区域 - 改为竖直布局
        gr.Markdown("### 📊 结果展示")
        
        with gr.Tabs():
            with gr.TabItem("📸 生成结果"):
                output_image = gr.Image(
                    label="生成的图像",
                    type="pil",
                    height=600,  # 增大显示高度
                    width=600
                )
                # 在生成结果下方添加重新编辑按钮
                with gr.Row():
                    continue_edit_btn = gr.Button("🔄 重新编辑", variant="primary", size="lg")
            
            with gr.TabItem("� 对比图"):
                comparison_image = gr.Image(
                    label="对比图 (原图|编辑图|生成结果)",
                    type="pil", 
                    height=400,  # 对比图稍小一些，因为是横向拼接的
                )
        
        # 示例区域
        gr.Markdown("### 💡 示例")
        gr.Examples(
            examples=[
                ["assets/vase.jpg", "A crystal vase with roses"],
                ["assets/room_corner.jpg", "A modern floor lamp"],
            ],
            inputs=[base_image, prompt],
            label="点击加载示例"
        )
        
        # 使用说明
        with gr.Accordion("📖 详细使用说明", open=False):
            gr.Markdown("""
            ### 步骤说明:
            1. **上传图像**: 选择你想要编辑的基础图像
            2. **在原图上编辑**: 
               - 使用**白色画笔**涂抹需要修复/替换的区域（mask区域）
               - 使用**黑色细画笔**在mask区域内勾勒你想要的内容轮廓
               - 完成后点击编辑器下方的"✅ 确认勾画"按钮
            3. **添加颜色提示**: 
               - 系统会自动将编辑结果同步到颜色编辑器
               - 在颜色编辑器中，使用**彩色画笔**在白色mask区域内添加颜色笔触
               - 颜色笔触会作为生成内容的颜色引导
            4. **设置颜色提示块数量**: 使用"🎯 颜色提示块数量"滑条选择要提取的颜色方块数量（1-70个）
            5. **生成颜色提示块**: 点击"🎨 确认生成颜色提示块"按钮
               - 系统会根据设定的数量自动提取纯色方块并保存为PNG文件
               - **文件路径会显示在界面上，可下载检查**
            6. **确认颜色提示**: 查看生成的条件图，确认无误后点击"✅ 确认颜色提示"
            7. **检查条件图**: 可通过"📥 下载条件图PNG"按钮下载检查最终输入给模型的条件图
            8. **输入prompt**: 详细描述你想在编辑区域生成的内容
            9. **调整参数**: 
               - 推理步数: 建议20-30，更多步数质量更好但速度更慢
               - 引导强度: 建议3-5，控制生成内容与prompt的相关性
            10. **生成图像**: 点击"🚀 生成图像"按钮开始处理
               - **模型将自动读取保存的PNG条件图文件**
            11. **重新编辑**: 生成完成后，点击"🔄 重新编辑"按钮可重新编辑
            
            ### 注意事项:
            - 确保已正确配置模型路径（在代码中修改local_path和lora_path）
            - 必须先用白笔涂抹区域，再用黑笔勾勒细节
            - **工作流程**: 确认勾画 → 生成颜色提示块 → 确认颜色提示 → 生成图像
            - **每一步都有明确的状态提示，确保不会卡死**
            - **按钮位置**: 确认勾画在编辑器下方，颜色按钮在颜色编辑器下方
            - 颜色提示会被提取为11x11像素的纯色方块
            - **条件图会自动保存为PNG文件到condition_images/目录**
            - **生成时模型直接读取PNG文件，确保输入一致性**
            - **可通过文件路径检查最终输入给模型的条件图**
            - 强制使用CUDA，确保GPU驱动正确安装
            
            ### 🚀 响应性问题解决方案:
            - **📊 查询状态**: 独立检查当前状态，不依赖网络同步
            - **✅ 确认勾画**: 极简版确认，减少网络依赖，自动同步到颜色编辑器
            - **🎨 生成颜色提示块**: 独立生成步骤，查看结果后再确认
            - **✅ 确认颜色提示**: 最终确认步骤，确保条件图正确
            - **🔄 重新编辑**: 生成后重置所有状态，流畅返回编辑模式
            - **异步保存**: 图片后台保存，不阻塞界面响应
            - **按钮位置优化**: 每个按钮紧跟对应的编辑器，操作更直观
            
            ### 📱 按钮使用指南:
            1. 编辑完成后 → 点击"✅ 确认编辑"
            2. 添加颜色后 → 调整"🎯 颜色提示块数量"滑条选择颜色块数量
            3. 生成颜色块 → 点击"🎨 确认生成颜色提示块"
            4. 查看条件图 → 检查生成的条件图是否正确
            5. 确认无误后 → 点击"✅ 确认颜色提示"
            6. 检查文件 → 点击"📥 下载条件图PNG"查看实际输入文件
            7. 生成完成后 → 点击"🔄 重新编辑"重新编辑
            8. 观察状态栏 → ✅表示已确认，⏳表示未确认
            
            ### 🌐 网络状态指示:
            - 🟢 绿色：正常 | 🟡 黄色：延迟 | 🔴 红色：异常
            - 状态实时更新，支持无缝编辑-生成-重新编辑循环
            - **分步确认设计，避免状态混乱和卡死**
            - **按钮布局优化，每个编辑器下方都有对应的确认按钮**
            """)
        
        # 事件绑定 - 添加更好的用户体验
        
        # 当上传新图像时，自动更新ImageEditor的背景
        def update_all_pads(base_img):
            """更新所有编辑器"""
            result = update_sketch_pad(base_img)
            return result, result
            
        base_image.change(
            fn=update_all_pads,
            inputs=base_image,
            outputs=[sketch_pad, color_pad],
            show_progress="hidden"
        )
        
        # 编辑区域变化监控 - 简化版本
        def check_sketch_and_network(sketch_data):
            """检查编辑变化并返回主状态字符串"""
            sketch_status = check_sketch_changes(sketch_data)
            return sketch_status

        sketch_pad.change(
            fn=check_sketch_and_network,
            inputs=sketch_pad,
            outputs=[status_text],
            show_progress="hidden"
        )
        
        # 重置按钮
        def reset_and_update_status(base_img):
            """重置并更新状态"""
            global edit_confirmed, color_confirmed, current_edit_data, current_color_data
            edit_confirmed = False
            color_confirmed = False
            current_edit_data = None
            current_color_data = None
            result = update_sketch_pad(base_img)
            # 返回：sketch_pad, color_pad, status_text, condition_preview
            return result, result, "🔄 已重置为原图，请重新编辑", None

        clear_btn.click(
            fn=reset_and_update_status,
            inputs=base_image,
            outputs=[sketch_pad, color_pad, status_text, condition_preview],
            show_progress="hidden"
        )
        
        
        # 状态查询按钮已移除; 使用主状态框 `status_text` 显示状态
        
        # 确认编辑按钮 - 简化版本
        def confirm_and_update_status_with_retry(base_img, sketch_data):
            """确认编辑并同步状态（简化返回）"""
            import time
            start_time = time.time()

            # 执行确认操作
            main_status, masked_image = confirm_edit_ready(base_img, sketch_data)

            if "✅" in main_status:
                # 更新颜色编辑器
                if masked_image is not None:
                    if isinstance(masked_image, Image.Image):
                        color_pad_image = np.array(masked_image)
                    else:
                        color_pad_image = masked_image
                else:
                    if base_img is not None and isinstance(base_img, Image.Image):
                        color_pad_image = np.array(base_img.resize((1024, 1024)))
                    else:
                        color_pad_image = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
            else:
                color_pad_image = np.ones((1024, 1024, 3), dtype=np.uint8) * 255

            response_time = time.time() - start_time
            print(f"⏱️ 确认响应: {response_time:.2f}秒, 状态: {'成功' if '✅' in main_status else '失败'}")

            return main_status, color_pad_image

        confirm_btn.click(
            fn=confirm_and_update_status_with_retry,
            inputs=[base_image, sketch_pad],
            outputs=[status_text, color_pad],
            show_progress="minimal"
        )
        
        # 生成颜色提示块按钮 - 使用slider值
        def generate_color_and_update_status(color_stroke_data, n_points_value):
            """生成颜色提示块并自动确认颜色（但不触发生成）"""
            global N_POINTS
            import time
            start_time = time.time()

            # 更新全局N_POINTS变量
            N_POINTS = int(n_points_value)
            print(f"🎯 使用颜色提示块数量: {N_POINTS}")

            # 生成颜色条件图并自动确认（内存中）
            color_msg, condition_img = generate_color_hints_from_strokes(color_stroke_data)
            response_time = time.time() - start_time
            print(f"⏱️ 颜色生成响应: {response_time:.2f}秒")

            # 返回：主状态（仅返回状态，条件图保存在内存current_color_data）
            # 返回状态文本和条件图以更新预览（条件图可能为None）
            return color_msg, condition_img

        confirm_generate_color_btn.click(
            fn=generate_color_and_update_status,
            inputs=[color_pad, n_points_slider],
            outputs=[status_text, condition_preview],
            show_progress="minimal"
        )
        
        # 确认颜色提示按钮 - 简化版本
        # 确认颜色提示按钮已移除; 颜色将在生成颜色提示块时自动确认
        
        # 下载条件图按钮
        # 条件图PNG下载功能已移除（使用内存中的条件图）
        
        # 生成按钮 - 增强状态反馈和同步
        def safe_generate_image_with_status(prompt_text, num_steps, guidance_scale):
            """安全的生成图像包装函数，确保始终有明确反馈"""
            import time
            timestamp = time.strftime("%H:%M:%S")
            print(f"🎯 [{timestamp}] 生成按钮被点击...")
            
            try:
                if not edit_confirmed:
                    error_msg = "❌ 请先点击'确认编辑完成'按钮"
                    return None, None, error_msg

                if not color_confirmed:
                    error_msg = "❌ 请先生成颜色提示块（颜色将在生成时自动确认）"
                    return None, None, error_msg

                print(f"⏳ [{timestamp}] 开始生成图像...")
                result_img, comparison_img, main_status = generate_image(prompt_text, num_steps, guidance_scale)

                return result_img, comparison_img, main_status

            except Exception as e:
                error_msg = f"❌ 生成过程出错: {str(e)}"
                print(f"❌ [{timestamp}] {error_msg}")
                return None, None, error_msg
        
        generate_event = generate_btn.click(
            fn=safe_generate_image_with_status,
            inputs=[prompt, num_steps, guidance_scale],
            outputs=[output_image, comparison_image, status_text],
            show_progress=True,
            scroll_to_output=True,
        )
        
        # 继续编辑按钮 - 重置状态并返回编辑模式
        def continue_editing_wrapper(base_img):
            """继续编辑的包装函数，确保返回值顺序正确"""
            sketch_pad_result, main_status, _, _, _, color_pad_result, _, _ = continue_editing(base_img)
            # 返回：sketch_pad, status_text, color_pad, condition_preview(清空)
            return sketch_pad_result, main_status, color_pad_result, None

        continue_edit_btn.click(
            fn=continue_editing_wrapper,
            inputs=base_image,
            outputs=[sketch_pad, status_text, color_pad, condition_preview],
            show_progress="hidden"
        )
        
        # 页面加载时初始化主状态文本
        def load_status_wrapper():
            main_status, *_ = get_current_status()
            return main_status

        demo.load(
            fn=load_status_wrapper,
            inputs=None,
            outputs=[status_text],
            show_progress="hidden"
        )
    
    return demo

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='OminiControl Inpainting Demo')
    parser.add_argument('--cpu', action='store_true', help='在CPU上运行（默认使用GPU）')
    parser.add_argument('--gpu', action='store_true', help='在GPU上运行（默认选项）')
    parser.add_argument('--port', type=int, default=7860, help='Gradio服务器端口（默认7860）')
    args = parser.parse_args()
    
    # 设置设备
    if args.cpu:
        USE_CPU = True
        DEVICE = "cpu"
        print("🚀 启动OminiControl Inpainting Demo (CPU模式)...")
        print("⚠️ CPU模式运行速度较慢，推荐使用GPU")
    else:
        # 检查CUDA可用性
        if torch.cuda.is_available():
            USE_CPU = False
            DEVICE = "cuda"
            print("🚀 启动OminiControl Inpainting Demo (GPU模式)...")
            print(f"✅ CUDA可用，GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CUDA不可用，自动切换到CPU模式")
            print("💡 如需使用GPU，请确保安装了正确的GPU驱动和CUDA版本")
            USE_CPU = True
            DEVICE = "cpu"
    
    # 创建输出目录
    os.makedirs("gradio_output", exist_ok=True)
    os.makedirs("condition_images", exist_ok=True)  # 创建条件图保存目录
    
    # 启动界面
    demo = create_ui()
    print(f"🌐 Gradio界面将在端口 {args.port} 启动")
    print(f"🎯 运行设备: {DEVICE}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        debug=True,
        show_error=True
    )
