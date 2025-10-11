# clean_cond_and_mask.py
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageOps

# --------------------- 基础 I/O ---------------------
def load_rgb(path):
    """读取图像、修正EXIF朝向；若带透明，先铺白底再转RGB"""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB")
    else:
        im = im.convert("RGB")
    return im

def load_mask_binary(path, size=None, small_comp_area=100):
    """
    读取mask并转为严格二值(0/255)：
    - 可选NEAREST缩放避免灰边
    - Otsu二值化
    - 开/闭运算去噪填洞
    - 去除小白色连通域(面积门限)
    """
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if size is not None:
        im = im.resize(size, Image.NEAREST)
    g = np.array(im.convert("L"))

    # Otsu二值化
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学净化
    kernel = np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)   # 去小噪
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)  # 填小孔

    # 去除很小的白色连通域（避免黑块中混入白点）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((bw==255).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(bw)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > small_comp_area:
            cleaned[labels==i] = 255
    return cleaned

# --------------------- 极性提示（仅日志） ---------------------
def polarity_hint(cond_rgb, mask_bw):
    """
    粗略判断 mask 白区是否更“彩色/有纹理”，仅用于日志提示（不自动翻转）
    你当前约定：白=线稿区域；若提示white_looks_color=True，可能数据极性相反
    """
    arr = np.array(cond_rgb).astype(np.float32) / 255.0
    sat_like = arr.std(axis=2)        # 通道方差≈彩度/纹理度
    m_white = (mask_bw == 255)
    m_black = ~m_white
    white_sat = float(sat_like[m_white].mean()) if m_white.any() else 0.0
    black_sat = float(sat_like[m_black].mean()) if m_black.any() else 0.0
    return {
        "white_sat": white_sat,
        "black_sat": black_sat,
        "white_looks_color": white_sat > black_sat
    }

# --------------------- 线稿净化（黑线纯黑、白底纯白） ---------------------
def clean_lineart_in_mask(cond_rgb, mask_bw,
                          black_th=60, white_th=240,
                          dilate_iters=1,
                          enforce_pure_white=True):
    """
    仅在 mask==255 的区域内：
      - 亮度 < black_th 视作“线条”→ 设为纯黑(0,0,0)，可选膨胀保持高分辨率线宽感知
      - 其它像素视作“白底”；若 enforce_pure_white=True，则设为纯白(255,255,255)
    返回：(净化后的PIL, 统计dict)
    """
    arr = np.array(cond_rgb).copy()
    m = (mask_bw == 255)

    # 亮度（避免彩色偏差）
    y = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.uint8)

    # 初始线条
    line0 = (y < black_th) & m

    # 可选膨胀（1024 vs 512建议 1~2）
    line = line0
    if dilate_iters > 0:
        kernel = np.ones((3,3), np.uint8)
        line = cv2.dilate(line.astype(np.uint8)*255, kernel, iterations=dilate_iters) > 127
        line = line & m

    # mask白区内的背景（非线条）
    bg = m & ~line

    # 白底纯度（净化前）
    not_pure_white_before = ((arr[bg][:,0] < 255) | (arr[bg][:,1] < 255) | (arr[bg][:,2] < 255)).mean() if bg.any() else 0.0
    near_white_before = ((y[bg] >= white_th)).mean() if bg.any() else 0.0

    # 写回：线→纯黑
    arr[line] = [0,0,0]

    # 写回：白底
    if enforce_pure_white:
        arr[bg] = [255,255,255]
    else:
        near_white = (y >= white_th) & bg
        arr[near_white] = [255,255,255]

    # 白底纯度（净化后）
    y2 = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.uint8)
    not_pure_white_after = ((arr[bg][:,0] < 255) | (arr[bg][:,1] < 255) | (arr[bg][:,2] < 255)).mean() if bg.any() else 0.0
    near_white_after = ((y2[bg] >= white_th)).mean() if bg.any() else 0.0

    stats = {
        "masked_area_ratio": float(m.mean()),
        "line_ratio_before": float(line0.mean()) if m.any() else 0.0,
        "line_ratio_after": float(line.mean()) if m.any() else 0.0,
        "bg_not_pure_white_before": float(not_pure_white_before),
        "bg_not_pure_white_after": float(not_pure_white_after),
        "bg_near_white_before": float(near_white_before),
        "bg_near_white_after": float(near_white_after),
        "thresholds": {"black_th": int(black_th), "white_th": int(white_th)},
        "dilate_iters": int(dilate_iters),
        "enforce_pure_white": bool(enforce_pure_white),
    }
    return Image.fromarray(arr), stats

# --------------------- 可视化叠加 ---------------------
def overlay_edges(rgb_img, mask_bw, color=(255,0,0)):
    arr = np.array(rgb_img).copy()
    edges = cv2.Canny(mask_bw, 0, 1)  # mask是0/255，阈值可极小
    arr[edges>0] = list(color)
    return Image.fromarray(arr)

# --------------------- 主流程 ---------------------
def process(cond_path, mask_path, out_dir="clean_out", out_size=None,
            resize_for_cond="NEAREST",  # "NEAREST"（线稿）或 "BICUBIC"（照片）
            black_th=60, white_th=240,
            dilate_iters=1,
            enforce_pure_white=True,
            small_comp_area=100):
    """
    cond_path: 条件图路径（线稿/白底/可能有彩色贴片）
    mask_path: mask路径（约定：白=线稿区域）
    out_size: 统一到的尺寸 (W,H), None=保持原尺寸
    resize_for_cond: 线稿建议 NEAREST；如果是照片背景可 BICUBIC
    """
    os.makedirs(out_dir, exist_ok=True)

    # 读取 cond
    cond = load_rgb(cond_path)
    if out_size is None:
        out_size = cond.size  # (W,H)
    pil_interp = Image.NEAREST if resize_for_cond.upper()=="NEAREST" else Image.BICUBIC
    cond = cond.resize(out_size, pil_interp)

    # 读取并标准化 mask（严格 0/255）
    mask = load_mask_binary(mask_path, size=out_size, small_comp_area=small_comp_area)

    # 极性提示（不自动翻转）
    pol = polarity_hint(cond, mask)

    # 净化线稿（仅在mask白区内：黑线→纯黑，白底→纯白）
    cond_clean, stats = clean_lineart_in_mask(
        cond, mask,
        black_th=black_th,
        white_th=white_th,
        dilate_iters=dilate_iters,
        enforce_pure_white=enforce_pure_white
    )

    # 可视化：mask白区中“非纯白”的像素（净化前后）
    arr_in  = np.array(cond)
    arr_out = np.array(cond_clean)
    m = (mask == 255)
    if m.any():
        non_pure_before = m & ((arr_in[:,:,0]<255) | (arr_in[:,:,1]<255) | (arr_in[:,:,2]<255))
        non_pure_after  = m & ((arr_out[:,:,0]<255) | (arr_out[:,:,1]<255) | (arr_out[:,:,2]<255))
    else:
        non_pure_before = np.zeros(mask.shape, dtype=bool)
        non_pure_after  = np.zeros(mask.shape, dtype=bool)

    viz_before = arr_in.copy()
    viz_before[non_pure_before] = [255, 0, 0]   # 红色标“白不纯”
    viz_after  = arr_out.copy()
    viz_after[non_pure_after]  = [255, 0, 0]

    # 输出文件
    Image.fromarray(mask).save(os.path.join(out_dir, "mask_binary.png"))
    cond.save(os.path.join(out_dir, "cond_input.png"))
    cond_clean.save(os.path.join(out_dir, "cond_clean.png"))
    overlay_edges(cond, mask).save(os.path.join(out_dir, "overlay_input_edges.png"))
    overlay_edges(cond_clean, mask).save(os.path.join(out_dir, "overlay_clean_edges.png"))
    Image.fromarray(viz_before).save(os.path.join(out_dir, "cond_non_pure_white_before.png"))
    Image.fromarray(viz_after ).save(os.path.join(out_dir, "cond_non_pure_white_after.png"))

    report = {
        "cond_path": cond_path,
        "mask_path": mask_path,
        "size_wh": out_size,
        "mask_white_ratio": float((mask==255).mean()),
        "polarity_hint": pol,     # 若 white_looks_color=True，可能极性相反（白不是线稿）
        "line_stats": stats,
        "notes": "mask 为严格0/255；在 mask=白 区域内将 cond 净化为 白底纯白 + 线条纯黑。"
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report

# --------------------- MAIN 示例 ---------------------
if __name__ == "__main__":
    # 把这两行换成你的实际路径
    COND_PATH = "./000000050.jpg"            # 线稿 + 白底 + 可能彩色贴片
    MASK_PATH = "./000000050.jpg.mask.png"   # 白=线稿区域

    # 运行处理
    rep = process(
        cond_path=COND_PATH,
        mask_path=MASK_PATH,
        out_dir="clean_out",
        out_size=(1024, 1024),       # 统一尺寸；保持原尺寸可设为 None
        resize_for_cond="NEAREST",   # 线稿建议 NEAREST；若是照片可改 "BICUBIC"
        black_th=60,                 # 线条阈值：越小越“严格黑”
        white_th=240,                # 近白阈值：用作统计与选择性净化
        dilate_iters=1,              # 512→1024 建议 1~2，让线宽视觉一致
        enforce_pure_white=True,     # 强制把 mask 白区的背景拉到 255
        small_comp_area=100          # 移除小白色连通域的面积阈值
    )
    print(json.dumps(rep, indent=2))
