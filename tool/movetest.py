import tkinter as tk
import mss
import time
import os
import shutil
import random
from datetime import datetime
import threading
from pynput import keyboard as pynput_keyboard
import numpy as np

# ========== 配置 ==========
TOTAL_TARGET = 120                # 目标张数
INTERVAL = 1.0                    # 间隔（秒/张）
WINDOW_POS = (100, 100)           # 悬浮窗位置

TEMP_DIR = "./s/f"
TRAIN_DIR = "./s/train"
VAL_DIR = "./s/val"
# ==========================

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# ---------- 截图区域坐标 ----------
# 原始角色可移动范围四个角（屏幕像素）
# 血条区域 = 可移动范围向上平移 180 像素（y 减 180）
# 再额外上移 20 像素（y 再减 20），总共上移 200 像素
screen_points = np.array([
    [136.5, 184.5],   # 左上  (384.5 - 200 = 184.5)
    [2392.5, 184.5],  # 右上
    [2532.5, 1074.5], # 右下  (1254.5 - 180 = 1074.5，只上移了180，因为上边界额外移了20，下边界不动)
    [11.5, 1074.5]    # 左下
], dtype=np.float32)

# 计算外接矩形（宽度保持不变，高度增加20）
min_x = int(np.min(screen_points[:, 0]))
max_x = int(np.max(screen_points[:, 0]))
min_y = int(np.min(screen_points[:, 1]))
max_y = int(np.max(screen_points[:, 1]))

region = {
    "left": min_x,
    "top": min_y,
    "width": max_x - min_x,
    "height": max_y - min_y
}
print(f"截图区域: 左上({min_x},{min_y}) 宽{region['width']} 高{region['height']}")
print(f"说明：上边界已向上额外移动 20 像素（总上移 200 像素），宽度未变。")

# ---------- 辅助函数：统计 TEMP_DIR 中实际图片数量 ----------
def get_image_count():
    if not os.path.exists(TEMP_DIR):
        return 0
    files = [f for f in os.listdir(TEMP_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return len(files)

class ScreenshotApp:
    def __init__(self):
        self.count = get_image_count()
        self.is_paused = True
        self.stop_flag = False

        # ---------- 透明悬浮窗 ----------
        self.root = tk.Tk()
        self.root.title("截图进度")
        self.root.geometry(f"320x100+{WINDOW_POS[0]}+{WINDOW_POS[1]}")
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'white')
        self.root.configure(bg='white')
        self.root.overrideredirect(True)

        self.label = tk.Label(
            self.root,
            text=f"【暂停】按 W 开始截图 (已有 {self.count} 张)",
            font=("微软雅黑", 16),
            fg="red",
            bg='white'
        )
        self.label.pack(expand=True)

        # 如果已经达到目标，直接分配
        if self.count >= TOTAL_TARGET:
            self.is_paused = True
            self.label.config(text=f"✅ 已达到 {self.count}/{TOTAL_TARGET}，正在分配...", fg="blue")
            self.root.update()
            self.split_dataset()
            self.label.config(text=f"✅ 完成！共 {self.count} 张，已按 8:2 分好", fg="blue")

        # ---------- 键盘监听 ----------
        self.listener = pynput_keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        # ---------- 截图线程 ----------
        if self.count < TOTAL_TARGET:
            self.thread = threading.Thread(target=self.screenshot_loop, daemon=True)
            self.thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_press(self, key):
        try:
            if key == pynput_keyboard.KeyCode.from_char('w') or key == pynput_keyboard.KeyCode.from_char('W'):
                self.toggle_pause()
        except:
            pass

    def toggle_pause(self):
        if self.count >= TOTAL_TARGET:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.label.config(text=f"【暂停】按 W 继续 (已有 {self.count} 张)", fg="red")
        else:
            self.label.config(text=f"截图中: {self.count}/{TOTAL_TARGET}", fg="green")

    def screenshot_loop(self):
        with mss.mss() as sct:
            while not self.stop_flag:
                current_count = get_image_count()
                self.count = current_count
                if self.count >= TOTAL_TARGET:
                    self.is_paused = True
                    self.root.after(0, self.finish_capture)
                    break

                if not self.is_paused and self.count < TOTAL_TARGET:
                    img = sct.grab(region)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(TEMP_DIR, f"screenshot_{ts}.png")
                    mss.tools.to_png(img.rgb, img.size, output=filename)
                    self.count = get_image_count()
                    self.root.after(0, self.update_label)
                    if self.count >= TOTAL_TARGET:
                        self.is_paused = True
                        self.root.after(0, self.finish_capture)
                        break
                time.sleep(INTERVAL)

    def update_label(self):
        if self.is_paused:
            self.label.config(text=f"已完成: {self.count}/{TOTAL_TARGET} (暂停)", fg="red")
        else:
            self.label.config(text=f"截图中: {self.count}/{TOTAL_TARGET}", fg="green")

    def finish_capture(self):
        self.label.config(text="✅ 正在划分数据集...", fg="blue")
        self.root.update()
        self.split_dataset()
        self.label.config(text=f"✅ 完成！共 {self.count} 张，已按 8:2 分好", fg="blue")
        self.is_paused = True

    def split_dataset(self):
        all_images = [f for f in os.listdir(TEMP_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not all_images:
            return
        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.8)
        train_set = all_images[:split_idx]
        val_set = all_images[split_idx:]

        for img in train_set:
            src = os.path.join(TEMP_DIR, img)
            dst = os.path.join(TRAIN_DIR, img)
            shutil.move(src, dst)

        for img in val_set:
            src = os.path.join(TEMP_DIR, img)
            dst = os.path.join(VAL_DIR, img)
            shutil.move(src, dst)

        try:
            os.rmdir(TEMP_DIR)
        except OSError:
            pass

    def on_close(self):
        self.stop_flag = True
        self.listener.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ScreenshotApp()
    app.run()