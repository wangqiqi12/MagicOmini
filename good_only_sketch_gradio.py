import os
import torch
import gradio as gr
import numpy as np
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageDraw
import cv2

from omini.pipeline.flux_omini import Condition, generate, seed_everything

# 全局变量
pipe = None
edit_confirmed = False  # 编辑确认状态
current_edit_data = None  # 当前编辑数据
last_sketch_hash = None  # 上次编辑数据的哈希值，用于检测变化
refresh_counter = 0  # 强制刷新计数器
local_backup = {}  # 本地状态备份，防止网络中断丢失状态
click_timestamps = []  # 点击时间戳，用于检测网络延迟
status_check_counter = 0  # 状态检查计数器
last_operation_time = 0  # 最后操作时间

def initialize_pipeline():
    """初始化pipeline - 强制使用CUDA，返回状态信息"""
    global pipe
    
    if pipe is not None:
        return True, "✅ Pipeline已就绪"
    
    print("🔄 正在初始化pipeline...")
    
    try:
        # 强制检查CUDA可用性
        if not torch.cuda.is_available():
            error_msg = "❌ CUDA不可用，请确保GPU驱动正确安装"
            print(error_msg)
            return False, error_msg
        
        print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
        
        # NOTE: 请修改为你的实际模型路径
        local_path = "/root/private_data/wangqiqi/Omini_ckpts/FLUX.1-dev"
        
        print(f"📂 加载基础模型: {local_path}")
        # 强制使用CUDA - 使用device_map自动处理设备放置
        pipe = FluxPipeline.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",  # 自动设备映射，会优先使用CUDA
            low_cpu_mem_usage=True,
        )
        
        # 验证模型设备信息
        try:
            if hasattr(pipe, 'device'):
                print(f"🔍 Pipeline设备: {pipe.device}")
            elif hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'device'):
                print(f"🔍 Transformer设备: {pipe.transformer.device}")
            else:
                print("🔍 设备信息: 使用device_map自动管理")
        except:
            print("🔍 设备信息: 自动管理中")
        
        print("📦 加载LoRA权重...")
        # NOTE: 请修改为你的实际LoRA路径  
        lora_path = "/root/private_data/wangqiqi/MagicOmini/runs/4GPU_bs1_acc8_tot32_1024_1024_r32_sketch_Prodigy/ckpt/15000"
        
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
        
        # 注意：使用device_map时不建议同时使用CPU卸载
        print("💡 使用device_map自动管理内存，跳过CPU卸载")
        
        print("🎉 Pipeline初始化完成!")
        return True, "✅ Pipeline初始化完成!"
        
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

def generate_image(prompt, num_steps, guidance_scale):
    """生成图像的主函数 - 使用确认的编辑数据"""
    global edit_confirmed, current_edit_data
    
    try:
        print(f"🔄 开始生成图像...")
        
        # 检查编辑确认状态
        if not edit_confirmed or current_edit_data is None:
            return None, None, "❌ 请先点击'确认编辑完成'按钮"
        
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
        
        print("🖼️ 使用已确认的编辑数据...")
        # 直接使用确认的masked图像
        try:
            masked_image = current_edit_data['masked_image']
            print("✅ 已确认的编辑数据加载成功")
        except Exception as mask_error:
            print(f"编辑数据错误: {mask_error}")
            return None, None, f"❌ 编辑数据加载失败: {str(mask_error)}"
        
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
            concat_image.paste(masked_image, (1024, 0))
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
    global edit_confirmed, current_edit_data, last_sketch_hash, status_check_counter, last_operation_time
    import time
    
    status_check_counter += 1
    current_time = time.time()
    timestamp = time.strftime("%H:%M:%S")
    
    # 强制刷新状态显示
    if edit_confirmed and current_edit_data:
        hash_info = current_edit_data.get('hash', 'unknown')
        main_status = f"✅ [{timestamp}] 编辑已确认 (哈希:{hash_info}) - 检查#{status_check_counter}"
        edit_status = "✅ 已确认，可以生成图像"
        network_status = "🟢 状态同步正常"
    else:
        main_status = f"⏳ [{timestamp}] 等待编辑确认 - 检查#{status_check_counter}"
        edit_status = "⏳ 请完成编辑并确认"
        network_status = f"🟡 未确认状态 - 检查#{status_check_counter}"
    
    print(f"📊 [{timestamp}] 状态查询#{status_check_counter}: confirmed={edit_confirmed}")
    return main_status, edit_status, network_status

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
    """继续编辑功能 - 重置所有状态并返回编辑模式"""
    global edit_confirmed, current_edit_data, last_sketch_hash
    import time
    
    timestamp = time.strftime("%H:%M:%S")
    print(f"🔄 [{timestamp}] 用户点击继续编辑，重置所有状态")
    
    # 重置所有编辑相关状态
    edit_confirmed = False
    current_edit_data = None
    last_sketch_hash = None
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 准备返回编辑模式的状态信息
    main_status = f"🔄 [{timestamp}] 已重置，请重新编辑和确认"
    edit_status = "⏳ 请完成编辑并确认"
    network_status = "🟢 状态已重置"
    
    # 更新编辑画布背景
    sketch_pad_result = update_sketch_pad(base_image)
    
    print(f"✅ [{timestamp}] 继续编辑状态重置完成")
    
    return sketch_pad_result, main_status, edit_status, network_status

def confirm_edit_ready(base_image, sketch_data):
    """确认编辑就绪 - 极简版本，专注状态设置"""
    global edit_confirmed, current_edit_data, last_sketch_hash, last_operation_time
    
    import time
    import hashlib
    current_time = time.time()
    timestamp = time.strftime("%H:%M:%S")
    
    print(f"🔄 [{timestamp}] ===== 开始确认编辑 =====")
    
    # 立即设置操作时间
    last_operation_time = current_time
    
    # 强制重置状态，确保每次都是新的确认
    edit_confirmed = False
    current_edit_data = None
    
    # 立即反馈用户操作已收到
    if base_image is None:
        print(f"❌ [{timestamp}] 基础图像为空")
        return "❌ 请先上传基础图像"
    
    if sketch_data is None:
        print(f"❌ [{timestamp}] 编辑数据为空")
        return "❌ 请先在图像上进行编辑"
    
    # 计算当前编辑数据的哈希值，用于检测变化
    try:
        if isinstance(sketch_data, np.ndarray):
            current_hash = hashlib.md5(sketch_data.tobytes()).hexdigest()[:8]
        else:
            current_hash = hashlib.md5(str(sketch_data).encode()).hexdigest()[:8]
        print(f"🔍 [{timestamp}] 编辑数据哈希: {current_hash}, 上次: {last_sketch_hash}")
    except:
        current_hash = "unknown"
    
    # 添加数据类型检查
    print(f"🔍 [{timestamp}] 检查数据类型 - base_image: {type(base_image)}, sketch_data: {type(sketch_data)}")
    
    try:
        # 快速验证编辑数据
        print(f"⏳ [{timestamp}] 开始处理编辑数据...")
        masked_image, status = create_masked_image_from_sketch(base_image, sketch_data)
        
        if masked_image is None:
            edit_confirmed = False
            print(f"❌ [{timestamp}] 编辑数据处理失败: {status}")
            return f"❌ 编辑数据无效: {status}"
        
        # 强制更新状态
        edit_confirmed = True
        last_sketch_hash = current_hash
        current_edit_data = {
            'base_image': base_image,
            'sketch_data': sketch_data,
            'masked_image': masked_image,
            'hash': current_hash,
            'timestamp': timestamp
        }
        
        # 立即备份成功状态
        backup_state()
        
        success_msg = f"✅ [{timestamp}] 编辑已确认！(哈希:{current_hash}) {status}"
        print(f"🎉 [{timestamp}] ===== 确认完成 ===== confirmed={edit_confirmed}")
        
        # 返回简洁明确的状态
        return success_msg
        
    except Exception as e:
        edit_confirmed = False
        error_msg = f"❌ [{timestamp}] 确认编辑失败: {str(e)}"
        print(error_msg)
        return error_msg



# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="OminiControl Inpainting Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 OminiControl Inpainting Demo")
        gr.Markdown("**使用说明**: 上传图像 → 编辑 → 确认 → 生成图像 | **响应慢?** 点击📊查询状态")
        
        # 竖直布局：上传图像区域
        with gr.Row():
            base_image = gr.Image(
                label="📤 1. 上传基础图像",
                type="pil",
                height=512,  # 增大显示高度
                width=512
            )
        
        # 编辑区域
        with gr.Row():
            sketch_pad = gr.ImageEditor(
                label="🖌️ 2. 在原图上编辑 (白笔涂抹mask区域，黑笔勾勒sketch)",
                type="numpy",
                height=600,  # 显著增大编辑区域
                width=600,
                brush=gr.Brush(
                    default_size=15,
                    colors=["#FFFFFF", "#000000"],  # 白色和黑色画笔
                    default_color="#FFFFFF"  # 默认白色（用于涂抹mask）
                ),
                value=np.ones((1024, 1024, 3), dtype=np.uint8) * 255  # 初始白色背景
            )
        
        # 控制按钮区域 
        with gr.Row():
            clear_btn = gr.Button("🗑️ 重置为原图", variant="secondary", size="sm")
            status_btn = gr.Button("📊 查询状态", variant="primary", size="sm")
            confirm_btn = gr.Button("✅ 确认编辑", variant="primary", size="sm")
        
        # 参数控制区域
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="✏️ 3. 输入Prompt描述",
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
        
        # 编辑状态监控
        with gr.Row():
            edit_status = gr.Textbox(
                label="🎯 编辑状态",
                value="⏳ 等待编辑...",
                interactive=False,
                lines=1
            )
        
        # 网络状态显示
        with gr.Row():
            network_status = gr.Textbox(
                label="🌐 网络状态",
                value="🟢 正常",
                interactive=False,
                lines=1
            )
        
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
                # 在生成结果下方添加继续编辑按钮
                with gr.Row():
                    continue_edit_btn = gr.Button("🔄 继续编辑", variant="primary", size="lg")
            
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
            3. **确认编辑**: 点击"✅ 确认编辑完成"按钮保存编辑数据
            4. **输入prompt**: 详细描述你想在编辑区域生成的内容
            5. **调整参数**: 
               - 推理步数: 建议20-30，更多步数质量更好但速度更慢
               - 引导强度: 建议3-5，控制生成内容与prompt的相关性
            6. **生成图像**: 点击"🚀 生成图像"按钮开始处理
            7. **继续编辑**: 生成完成后，点击"🔄 继续编辑"按钮重新编辑
            
            ### 注意事项:
            - 确保已正确配置模型路径（在代码中修改local_path和lora_path）
            - 必须先用白笔涂抹区域，再用黑笔勾勒细节
            - **必须点击"确认编辑完成"按钮**才能进行生成
            - 编辑后的图像（包含白色mask和黑色sketch）将作为条件图输入模型
            - 强制使用CUDA，确保GPU驱动正确安装
            
            ### 🚀 响应性问题解决方案:
            - **📊 查询状态**: 独立检查当前状态，不依赖网络同步
            - **✅ 确认编辑**: 极简版确认，减少网络依赖
            - **🔄 继续编辑**: 生成后重置状态，流畅返回编辑模式
            - **异步保存**: 图片后台保存，不阻塞界面响应
            
            ### 📱 按钮使用指南:
            1. 编辑完成后 → 点击"✅ 确认编辑"
            2. 如果没反应 → 点击"📊 查询状态"查看当前状态  
            3. 生成完成后 → 点击"🔄 继续编辑"重新编辑
            4. 观察状态栏 → ✅表示已确认，⏳表示未确认
            
            ### 🌐 网络状态指示:
            - 🟢 绿色：正常 | 🟡 黄色：延迟 | 🔴 红色：异常
            - 状态实时更新，支持无缝编辑-生成-继续编辑循环
            """)
        
        # 事件绑定 - 添加更好的用户体验
        
        # 当上传新图像时，自动更新ImageEditor的背景
        base_image.change(
            fn=update_sketch_pad,
            inputs=base_image,
            outputs=sketch_pad,
            show_progress="hidden"
        )
        
        # 编辑区域变化监控 - 检测编辑状态变化，带网络状态更新
        def check_sketch_and_network(sketch_data):
            """检查编辑变化和网络状态"""
            sketch_status = check_sketch_changes(sketch_data)
            network_msg = check_network_status()
            return sketch_status, network_msg
        
        sketch_pad.change(
            fn=check_sketch_and_network,
            inputs=sketch_pad,
            outputs=[edit_status, network_status],
            show_progress="hidden"
        )
        
        # 重置按钮
        def reset_and_update_status(base_img):
            """重置并更新状态"""
            global edit_confirmed, current_edit_data
            edit_confirmed = False
            current_edit_data = None
            result = update_sketch_pad(base_img)
            return result, "🔄 已重置为原图，请重新编辑"
        
        clear_btn.click(
            fn=reset_and_update_status,
            inputs=base_image,
            outputs=[sketch_pad, edit_status],
            show_progress="hidden"
        )
        
        # 网络状态检测
        def check_network_status():
            """检测网络状态和响应延迟"""
            import time
            global click_timestamps
            
            current_time = time.time()
            
            # 计算最近点击的响应延迟
            if len(click_timestamps) >= 2:
                avg_delay = sum(click_timestamps[-3:]) / len(click_timestamps[-3:]) - click_timestamps[-1]
                if avg_delay > 2.0:
                    return "🔴 网络延迟较高，建议点击刷新按钮"
                elif avg_delay > 1.0:
                    return "🟡 网络有轻微延迟"
            
            return "🟢 网络状态正常"
        # 状态查询按钮 - 独立状态检查，不依赖网络同步
        status_btn.click(
            fn=get_current_status,
            inputs=None,
            outputs=[status_text, edit_status, network_status],
            show_progress="hidden"
        )
        
        # 确认编辑按钮 - 网络容错版本
        def confirm_and_update_status_with_retry(base_img, sketch_data):
            """确认编辑并同步状态 - 带网络状态检测"""
            import time
            start_time = time.time()
            
            # 执行确认操作
            main_status = confirm_edit_ready(base_img, sketch_data)
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            if "✅" in main_status:
                edit_status_msg = "✅ 编辑已确认，可以生成图像"
                if response_time > 3.0:
                    network_msg = "🟡 响应较慢，已完成确认"
                else:
                    network_msg = "🟢 确认成功，响应正常"
            else:
                edit_status_msg = "❌ 编辑确认失败，请重试或点击刷新"
                if response_time > 5.0:
                    network_msg = "🔴 网络超时，建议点击刷新按钮"
                else:
                    network_msg = "🟡 确认失败，网络正常"
            
            print(f"⏱️ 确认操作响应时间: {response_time:.2f}秒")
            return main_status, edit_status_msg, network_msg
        
        confirm_btn.click(
            fn=confirm_and_update_status_with_retry,
            inputs=[base_image, sketch_pad],
            outputs=[status_text, edit_status, network_status],
            show_progress="minimal"
        )
        
        # 生成按钮 - 增强状态反馈和同步
        def safe_generate_image_with_status(prompt_text, num_steps, guidance_scale):
            """安全的生成图像包装函数，确保始终有明确反馈"""
            import time
            timestamp = time.strftime("%H:%M:%S")
            print(f"🎯 [{timestamp}] 生成按钮被点击...")
            
            try:
                if not edit_confirmed:
                    error_msg = "❌ 请先点击'确认编辑完成'按钮"
                    return None, None, error_msg, "❌ 未确认编辑，无法生成"
                
                print(f"⏳ [{timestamp}] 开始生成图像...")
                result_img, comparison_img, main_status = generate_image(prompt_text, num_steps, guidance_scale)
                
                # 生成完成后的状态
                if result_img is not None:
                    edit_status_msg = "🎉 生成完成！如需重新编辑请修改后重新确认"
                else:
                    edit_status_msg = "❌ 生成失败，请检查编辑数据"
                
                return result_img, comparison_img, main_status, edit_status_msg
                
            except Exception as e:
                error_msg = f"❌ 生成过程出错: {str(e)}"
                print(f"❌ [{timestamp}] {error_msg}")
                return None, None, error_msg, "❌ 生成过程异常"
        
        generate_event = generate_btn.click(
            fn=safe_generate_image_with_status,
            inputs=[prompt, num_steps, guidance_scale],
            outputs=[output_image, comparison_image, status_text, edit_status],
            show_progress=True,
            scroll_to_output=True,
        )
        
        # 继续编辑按钮 - 重置状态并返回编辑模式
        continue_edit_btn.click(
            fn=continue_editing,
            inputs=base_image,
            outputs=[sketch_pad, status_text, edit_status, network_status],
            show_progress="hidden"
        )
        
        # 页面加载时初始化状态
        demo.load(
            fn=get_current_status,
            inputs=None,
            outputs=[status_text, edit_status, network_status],
            show_progress="hidden"
        )
    
    return demo

if __name__ == "__main__":
    print("🚀 启动OminiControl Inpainting Demo...")
    
    # 强制检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✅ CUDA可用，GPU: {torch.cuda.get_device_name()}")
    else:
        print("❌ CUDA不可用，程序将无法正常工作")
        print("💡 请确保安装了正确的GPU驱动和CUDA版本")
        exit(1)
    
    # 创建输出目录
    os.makedirs("gradio_output", exist_ok=True)
    
    # 启动界面
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
