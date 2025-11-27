from PIL import Image
import numpy as np

def make_gif_crossfade(image_paths, out_path="out_crossfade.gif",
                       hold_frames=6, fade_frames=8, duration=40,
                       line_width=4):

    """
    右 → 左 渐变替换 (Wipe) + 黑色边界线
    line_width: 黑线宽度
    """

    # ====== 读取图片（切三份）======
    imgs = Image.open(image_paths).convert("RGB")
    tmp_img = []
    for i in range(3):
        tmp_img.append(imgs.crop((i * imgs.width // 3, 0, (i + 1) * imgs.width // 3, imgs.height)))
    imgs = tmp_img

    w, h = imgs[0].size

    frames = []
    n = len(imgs)

    for i in range(n):
        a = imgs[i]                     # 当前图
        b = imgs[(i + 1) % n]           # 下一图

        # 保持当前图
        for _ in range(hold_frames):
            frames.append(a.copy())

        # ====== 右 → 左 Wipe 替换 ======
        for t in range(1, fade_frames + 1):
            frame = Image.new("RGB", (w, h))

            ratio = t / fade_frames
            cut_x = int((1 - ratio) * w)   # 推进边界位置（右→左）

            # 左边：下一张图 b
            if cut_x < w:
                b_crop = b.crop((cut_x, 0, w, h))
                frame.paste(b_crop, (cut_x, 0))

            # 右边：当前图 a
            if cut_x > 0:
                a_crop = a.crop((0, 0, cut_x, h))
                frame.paste(a_crop, (0, 0))

            # ====== 添加黑色边界线 ======
            if 0 < cut_x < w:
                line_left = max(0, cut_x - line_width // 2)
                line_right = min(w, cut_x + line_width // 2)

                black_line = Image.new("RGB", (line_right - line_left, h), (0, 0, 0))
                frame.paste(black_line, (line_left, 0))

            frames.append(frame)

    # ====== 保存 GIF ======
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        disposal=2
    )


if __name__ == "__main__":
    make_gif_crossfade(
        "./tall building.webp",
        out_path="demo_crossfade.gif",
        hold_frames=5,
        fade_frames=20,
        duration=40,
        line_width=6   # 👉 调整黑线宽度
    )
