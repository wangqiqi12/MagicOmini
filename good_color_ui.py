import gradio as gr
import numpy as np
import cv2
import os
from datetime import datetime

def extract_color_hints_from_strokes(stroke_image, original_cond_image, radius=5, n_points=30):
    """从颜色笔触中直接提取纯色方块 - 参考UI_app.py的generate_color_hints_like_reference函数"""
    if stroke_image is None or original_cond_image is None:
        return None
        
    h, w = stroke_image.shape[:2]
    
    # 检测原始条件图中的白色mask区域
    original_gray = cv2.cvtColor(original_cond_image, cv2.COLOR_RGB2GRAY)
    white_mask_area = original_gray > 240  # 白色区域
    
    if not np.any(white_mask_area):
        print("Debug: No white mask area found in original condition image")
        return original_cond_image.copy()
    
    # 检查图像形状是否匹配
    if stroke_image.shape != original_cond_image.shape:
        print(f"Debug: Shape mismatch - stroke: {stroke_image.shape}, original: {original_cond_image.shape}")
        return original_cond_image.copy()
    
    # 计算编辑前后的差异，找到新添加的颜色stroke - 参考UI_app.py的逻辑
    diff = np.abs(stroke_image.astype(np.float32) - original_cond_image.astype(np.float32))
    diff_sum = np.sum(diff, axis=2)
    
    # 检测有明显变化的区域
    significant_change = diff_sum > 30
    
    # 检测stroke_image中的颜色（排除黑色和白色）
    stroke_gray = cv2.cvtColor(stroke_image, cv2.COLOR_RGB2GRAY)
    has_color = (stroke_gray > 50) & (stroke_gray < 240)  # 不是黑色也不是白色
    
    # 找到既在白色mask区域、又有颜色、又是新添加的像素
    valid_color_indices = np.argwhere(significant_change & has_color & white_mask_area)
    
    if len(valid_color_indices) == 0:
        print("Debug: No valid color strokes found in white mask area")
        return original_cond_image.copy()
    
    print(f"Debug: Found {len(valid_color_indices)} valid color stroke pixels")
    
    # 创建新的条件图，从原图开始
    new_cond_image = original_cond_image.copy()
    
    # 从有颜色的像素中随机采样 - 参考UI_app.py
    n_sample = min(n_points, len(valid_color_indices))
    sampled_indices = valid_color_indices[np.random.choice(
        len(valid_color_indices), size=n_sample, replace=False)]
    
    print(f"Debug: Sampling {n_sample} color points from {len(valid_color_indices)} candidates")
    
    n_valid = 0
    for y, x in sampled_indices:
        # 边界检查
        if y - radius < 0 or y + radius >= h or x - radius < 0 or x + radius >= w:
            continue
        
        # 检查这个patch是否完全在白色mask内 - 参考UI_app.py
        patch_white_area = white_mask_area[y - radius:y + radius + 1, x - radius:x + radius + 1]
        if patch_white_area.shape != (2 * radius + 1, 2 * radius + 1):
            continue
        if not np.all(patch_white_area):
            continue  # 不完全在白色区域内，跳过
        
        # 获取颜色并设置color hint - 与UI_app.py保持一致
        raw_color = stroke_image[y, x]
        
        # 确保颜色值不为白色（255,255,255）
        if np.all(raw_color >= 248):
            # 如果颜色太接近白色，稍微调暗
            color = np.clip(raw_color.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
        else:
            color = raw_color
            
        # 重要：创建一个固定纯色值，确保整个方块都是完全相同的颜色
        fixed_color = [int(color[0]), int(color[1]), int(color[2])]  # 转换为Python整数
        
        # 填充方块区域，每个像素都是完全相同的纯色
        block_height = 2 * radius + 1
        block_width = 2 * radius + 1
        color_block = np.full((block_height, block_width, 3), fixed_color, dtype=np.uint8)
        
        # 将整个纯色块赋值到目标区域
        new_cond_image[y - radius:y + radius + 1, x - radius:x + radius + 1] = color_block
        
        # 确保方块内每个像素都是完全相同的纯色（不计算均值，直接设置为fixed_color）
        new_cond_image[y - radius:y + radius + 1, x - radius:x + radius + 1] = fixed_color
        
        n_valid += 1
        print(f"Debug: Added color hint block at ({x}, {y}) with color {fixed_color}")
        
        if n_valid >= n_points:
            break  # 达到期望数量就停止
    
    print(f"Debug: Generated {n_valid} color hint points from user strokes")
    return new_cond_image

def ensure_rgb_format(image):
    """确保图像是RGB格式（3通道）"""
    if image is None:
        return None
    
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA格式
            return image[:, :, :3]  # 转换为RGB
        elif image.shape[2] == 3:  # 已经是RGB格式
            return image
    elif len(image.shape) == 2:  # 灰度图
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    return image

def process_color_hints(original_cond, stroke_edited):
    """处理颜色提示并生成新的条件图"""
    try:
        if original_cond is None:
            return None, "❌ 请先上传条件图"
        
        if stroke_edited is None:
            return original_cond, "⚠️ 未检测到编辑，返回原图"
        
        print(f"Debug: original_cond shape: {original_cond.shape}")
        print(f"Debug: stroke_edited type: {type(stroke_edited)}")
        
        # 确保图像格式一致
        original_cond = ensure_rgb_format(original_cond)
        
        # 处理ImageEditor返回的数据格式
        if isinstance(stroke_edited, dict):
            print(f"Debug: stroke_edited keys: {stroke_edited.keys()}")
            if 'composite' in stroke_edited and stroke_edited['composite'] is not None:
                stroke_image = stroke_edited['composite']
                print(f"Debug: Got composite image with shape: {stroke_image.shape}")
            else:
                return original_cond, "⚠️ 未检测到有效编辑，返回原图"
        else:
            stroke_image = stroke_edited
        
        stroke_image = ensure_rgb_format(stroke_image)
        
        if stroke_image is None or original_cond is None:
            return None, "❌ 图像格式错误"
        
        # 检查图像尺寸是否匹配
        if stroke_image.shape != original_cond.shape:
            return None, f"❌ 图像尺寸不匹配: {stroke_image.shape} vs {original_cond.shape}"
        
        print(f"Debug: Processing images with shape: {stroke_image.shape}")
        
        # 提取颜色提示并生成新的条件图
        new_cond_image = extract_color_hints_from_strokes(stroke_image, original_cond, radius=5, n_points=30)
        
        if new_cond_image is None:
            return original_cond, "⚠️ 颜色提示提取失败，返回原图"
        
        # 检查是否有实际的变化
        if np.array_equal(new_cond_image, original_cond):
            return original_cond, "⚠️ 未检测到颜色变化，请在白色区域添加颜色笔触"
        
        return new_cond_image, "✅ 颜色提示已添加到条件图中"
        
    except Exception as e:
        print(f"Error in process_color_hints: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ 处理过程中出错: {str(e)}"

def save_result_image(image):
    """保存结果图像为PNG格式"""
    if image is None:
        return None, "❌ 没有图像可以保存"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    
    try:
        # 保存为PNG格式，确保无损
        filename = f"output/cond_{timestamp}.png"
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), png_params)
        
        return filename, f"✅ 图像已保存为: {filename}"
    except Exception as e:
        return None, f"❌ 保存失败: {str(e)}"

# 创建Gradio界面
with gr.Blocks(title="🎨 颜色提示添加器") as demo:
    gr.Markdown("# 🎨 颜色提示添加器")
    gr.Markdown("""
    ### 📝 使用说明：
    1. **上传条件图** - 上传包含白色mask区域和黑色sketch的条件图
    2. **添加颜色** - 在白色区域内用画笔添加颜色笔触
    3. **生成结果** - 系统自动提取颜色方块并生成新的条件图
    4. **下载结果** - 点击下载按钮保存最终结果
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # 上传原始条件图
            gr.Markdown("## 📤 上传条件图")
            original_cond = gr.Image(
                label="上传包含白色mask和黑色sketch的条件图",
                type="numpy",
                height=300
            )
            
            # 控制按钮
            gr.Markdown("## 🔧 操作")
            generate_btn = gr.Button("🎯 生成新条件图", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ 清空", variant="secondary")
            
            # 状态信息
            status_text = gr.Textbox(
                label="状态信息", 
                interactive=False, 
                value="请上传条件图并开始编辑"
            )
        
        with gr.Column(scale=2):
            # 编辑区域
            gr.Markdown("## 🎨 添加颜色提示")
            color_editor = gr.ImageEditor(
                label="在白色区域内添加颜色笔触",
                type="numpy",
                height=400
            )
    
    # 结果展示
    gr.Markdown("## 📸 结果")
    with gr.Row():
        result_image = gr.Image(
            label="🎯 新的条件图",
            type="numpy",
            height=400
        )
        download_file = gr.File(
            label="📥 下载结果"
        )
    
    # 事件处理函数
    def update_editor(original):
        """上传图片时更新编辑器"""
        if original is None:
            return None, "请上传条件图"
        
        original = ensure_rgb_format(original)
        if original is None:
            return None, "图片格式错误，请重新上传"
        
        return original, "条件图已上传，请在白色区域添加颜色"
    
    def generate_new_cond(original, edited):
        """生成新的条件图"""
        new_image, status = process_color_hints(original, edited)
        
        if new_image is not None:
            # 保存文件
            filepath, save_status = save_result_image(new_image)
            if filepath:
                return new_image, status + "\n" + save_status, filepath
            else:
                return new_image, status + "\n" + save_status, None
        else:
            return None, status, None
    
    def clear_all():
        """清空所有内容"""
        return None, None, None, "已清空所有内容", None
    
    # 事件绑定
    original_cond.change(
        fn=update_editor,
        inputs=[original_cond],
        outputs=[color_editor, status_text]
    )
    
    generate_btn.click(
        fn=generate_new_cond,
        inputs=[original_cond, color_editor],
        outputs=[result_image, status_text, download_file]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[original_cond, color_editor, result_image, status_text, download_file]
    )

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7861)
