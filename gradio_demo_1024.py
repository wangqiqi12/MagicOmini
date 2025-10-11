import os
import torch
import gradio as gr
import numpy as np
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageDraw
import cv2

from omini.pipeline.flux_omini import Condition, generate, seed_everything

# 全局变量存储pipeline
pipe = None

def initialize_pipeline():
    """初始化pipeline"""
    global pipe
    if pipe is None:
        print("正在初始化pipeline...")
        try:
            # NOTE: 请修改为你的实际模型路径
            local_path = "/root/private_data/wangqiqi/Omini_ckpts/FLUX.1-dev"
            pipe = FluxPipeline.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16
            )
            pipe = pipe.to("cuda")
            
            # NOTE: 请修改为你的实际LoRA路径  
            lora_path = "/root/private_data/wangqiqi/Omini_ckpts/lora_sketch_1024_1024_5w"
            pipe.load_lora_weights(
                lora_path,
                weight_name="default.safetensors",
                adapter_name="sketch",
            )
            print("Pipeline初始化完成!")
            return True
        except Exception as e:
            print(f"Pipeline初始化失败: {e}")
            return False
    return True

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

def generate_image(base_image, sketch_data, prompt, num_steps, guidance_scale):
    """生成图像的主函数"""
    try:
        # 初始化pipeline
        if not initialize_pipeline():
            return None, None, "Pipeline初始化失败，请检查模型路径"
        
        if not prompt.strip():
            return None, None, "请输入prompt描述"
        
        # 创建masked图像
        masked_image, mask_status = create_masked_image_from_sketch(base_image, sketch_data)
        
        if masked_image is None:
            return None, None, mask_status
        
        # 创建condition
        # 创建condition前激活 LoRA
        pipe.set_adapters("sketch")  # ← 新增

        condition = Condition(masked_image, "sketch")
        
        # 设置随机种子
        seed_everything(42)
        
        # 生成图像
        print(f"正在生成图像，prompt: {prompt}")
        result = generate(
            pipe,
            prompt=prompt,
            conditions=[condition],
            height=1024,
            width=1024,
            num_inference_steps=int(num_steps),
            guidance_scale=guidance_scale,
        )
        
        result_img = result.images[0]
        
        # 创建对比图像
        concat_image = Image.new("RGB", (1024 * 3, 1024))
        base_resized = base_image.resize((1024, 1024)) if base_image else Image.new("RGB", (1024, 1024), (255, 255, 255))
        concat_image.paste(base_resized, (0, 0))
        concat_image.paste(masked_image, (1024, 0))
        concat_image.paste(result_img, (1024 * 2, 0))
        
        # 保存结果
        os.makedirs("gradio_output", exist_ok=True)
        result_img.save("gradio_output/result.jpg")
        concat_image.save("gradio_output/comparison.jpg")
        
        return result_img, concat_image, "生成成功！"
        
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        print(error_msg)
        return None, None, error_msg

def update_sketch_pad(base_image):
    """当上传新图像时，更新绘制画布的背景"""
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

def preview_mask(base_image, sketch_data):
    """预览编辑效果"""
    if base_image is None:
        return None, "请先上传基础图像"
    
    masked_image, status = create_masked_image_from_sketch(base_image, sketch_data)
    return masked_image, status

# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="OminiControl Inpainting Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 OminiControl Inpainting Demo")
        gr.Markdown("**使用说明**: 上传图像 → 在原图上用白笔涂抹mask区域 → 用黑笔勾勒sketch → 生成图像")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                base_image = gr.Image(
                    label="📤 1. 上传基础图像",
                    type="pil",
                    height=250
                )
                
                sketch_pad = gr.ImageEditor(
                    label="🖌️ 2. 在原图上编辑 (白笔涂抹mask区域，黑笔勾勒sketch)",
                    type="numpy",
                    height=300,
                    brush=gr.Brush(
                        default_size=15,
                        colors=["#FFFFFF", "#000000"],  # 白色和黑色画笔
                        default_color="#FFFFFF"  # 默认白色（用于涂抹mask）
                    ),
                    value=np.ones((1024, 1024, 3), dtype=np.uint8) * 255  # 初始白色背景
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 重置为原图", variant="secondary")
                    preview_btn = gr.Button("👁️ 预览编辑效果", variant="secondary")
                
                prompt = gr.Textbox(
                    label="✏️ 3. 输入Prompt描述",
                    placeholder="描述你想要在mask区域生成的内容，例如：A beautiful flower vase",
                    lines=3,
                    value="A beautiful vase"
                )
                
                with gr.Row():
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=28,
                        step=1,
                        label="推理步数"
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.5,
                        step=0.1,
                        label="引导强度"
                    )
                
                generate_btn = gr.Button("🚀 生成图像", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # 输出区域
                gr.Markdown("### 📊 结果展示")
                
                with gr.Tabs():
                    with gr.TabItem("生成结果"):
                        output_image = gr.Image(
                            label="生成的图像",
                            type="pil",
                            height=300
                        )
                    
                    with gr.TabItem("对比图"):
                        comparison_image = gr.Image(
                            label="对比图 (原图|Mask|生成结果)",
                            type="pil",
                            height=300
                        )
                    
                    with gr.TabItem("编辑效果"):
                        mask_preview = gr.Image(
                            label="编辑效果预览",
                            type="pil",
                            height=300
                        )
                
                status_text = gr.Textbox(
                    label="📝 状态信息",
                    value="请上传图像并绘制mask区域",
                    interactive=False,
                    lines=2
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
            3. **预览编辑**: 点击"预览编辑效果"查看编辑后的图像
            4. **输入prompt**: 详细描述你想在编辑区域生成的内容
            5. **调整参数**: 
               - 推理步数: 建议20-30，更多步数质量更好但速度更慢
               - 引导强度: 建议3-5，控制生成内容与prompt的相关性
            6. **生成图像**: 点击生成按钮开始处理
            
            ### 注意事项:
            - 确保已正确配置模型路径（在代码中修改local_path和lora_path）
            - 先用白笔涂抹区域，再用黑笔勾勒细节
            - 编辑后的图像（包含白色mask和黑色sketch）将作为条件图输入模型
            """)
        
        # 事件绑定
        # 当上传新图像时，自动更新ImageEditor的背景
        base_image.change(
            fn=update_sketch_pad,
            inputs=base_image,
            outputs=sketch_pad
        )
        
        clear_btn.click(
            fn=update_sketch_pad,  # 重置为原图
            inputs=base_image,
            outputs=sketch_pad
        )
        
        preview_btn.click(
            fn=preview_mask,
            inputs=[base_image, sketch_pad],
            outputs=[mask_preview, status_text]
        )
        
        generate_btn.click(
            fn=generate_image,
            inputs=[base_image, sketch_pad, prompt, num_steps, guidance_scale],
            outputs=[output_image, comparison_image, status_text],
            show_progress=True
        )
    
    return demo

if __name__ == "__main__":
    print("🚀 启动OminiControl Inpainting Demo...")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✅ CUDA可用，GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ CUDA不可用，将使用CPU（速度会很慢）")
    
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
