import tkinter as tk
import mss
import time
import os
import shutil
import random
from datetime import datetime
import threading
from pynput import keyboard as pynput_keyboard

# ========== 配置 ==========
SAVE_DIR = "screenshots"          # 临时存放原始截图
TOTAL_TARGET = 120                # 目标张数
INTERVAL = 1.0                    # 间隔（秒/张）
WINDOW_POS = (100, 100)           # 悬浮窗位置
DATASET_ROOT = "dataset"          # 数据集根目录
# ==========================

# 创建必要的文件夹
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{DATASET_ROOT}/images/train", exist_ok=True)
os.makedirs(f"{DATASET_ROOT}/images/val", exist_ok=True)
os.makedirs(f"{DATASET_ROOT}/labels/train", exist_ok=True)
os.makedirs(f"{DATASET_ROOT}/labels/val", exist_ok=True)

class ScreenshotApp:
    def __init__(self):
        self.count = 0
        self.is_paused = True          # 初始暂停
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
            text="【暂停】按 Q 开始截图",
            font=("微软雅黑", 18),
            fg="red",
            bg='white'
        )
        self.label.pack(expand=True)

        # ---------- 键盘监听（pynput，游戏内有效） ----------
        self.listener = pynput_keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        # ---------- 截图线程 ----------
        self.thread = threading.Thread(target=self.screenshot_loop, daemon=True)
        self.thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_press(self, key):
        try:
            if key == pynput_keyboard.KeyCode.from_char('q') or key == pynput_keyboard.KeyCode.from_char('Q'):
                self.toggle_pause()
        except:
            pass

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.label.config(text="【暂停】按 Q 继续", fg="red")
        else:
            self.label.config(text=f"截图中: {self.count}/{TOTAL_TARGET}", fg="green")

    def screenshot_loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while not self.stop_flag:
                if not self.is_paused and self.count < TOTAL_TARGET:
                    img = sct.grab(monitor)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(SAVE_DIR, f"screenshot_{ts}.png")
                    mss.tools.to_png(img.rgb, img.size, output=filename)
                    self.count += 1
                    self.root.after(0, self.update_label)
                    if self.count >= TOTAL_TARGET:
                        self.is_paused = True
                        self.root.after(0, self.finish_capture)
                time.sleep(INTERVAL)

    def update_label(self):
        if self.is_paused:
            self.label.config(text=f"已完成: {self.count}/{TOTAL_TARGET} (暂停)", fg="red")
        else:
            self.label.config(text=f"截图中: {self.count}/{TOTAL_TARGET}", fg="green")

    def finish_capture(self):
        self.label.config(text="✅ 正在划分数据集...", fg="blue")
        self.root.update()
        # 划分数据集
        self.split_dataset()
        self.label.config(text=f"✅ 完成！共 {self.count} 张，已按 8:2 分好", fg="blue")
        self.is_paused = True

    def split_dataset(self):
        """随机将 screenshots 中的图片按 8:2 移动到 dataset/images/train 和 val"""
        all_images = [f for f in os.listdir(SAVE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not all_images:
            return
        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.8)
        train_set = all_images[:split_idx]
        val_set = all_images[split_idx:]

        for img in train_set:
            src = os.path.join(SAVE_DIR, img)
            dst = os.path.join(DATASET_ROOT, "images", "train", img)
            shutil.move(src, dst)

        for img in val_set:
            src = os.path.join(SAVE_DIR, img)
            dst = os.path.join(DATASET_ROOT, "images", "val", img)
            shutil.move(src, dst)

        # 删除空文件夹（可选）
        if not os.listdir(SAVE_DIR):
            os.rmdir(SAVE_DIR)

    def on_close(self):
        self.stop_flag = True
        self.listener.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ScreenshotApp()
    app.run()