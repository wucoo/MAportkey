import sys
import cv2
import numpy as np
import mss
from ultralytics import YOLO
import time
import math
import pyautogui
import keyboard
import torch
import json
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMetaObject, pyqtSlot, Q_ARG
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush

# ---------- 配置 ----------
MODEL_PATH = "./best.pt"
CONF_THRESHOLD = 0.3          # 降低阈值，提高召回
CLASS_NAMES = {0: 'mb', 1: 'eb'}

# ---------- 可调参数（默认值） ----------
FOOT_OFFSET_Y = 210
OFFSET_X = 0
OFFSET_Y = 0

# ---------- 配置文件路径 ----------
CONFIG_DIR = "./set"
CONFIG_FILE = os.path.join(CONFIG_DIR, "opt.json")

# ---------- 配置管理 ----------
def load_config():
    global CONF_THRESHOLD, FOOT_OFFSET_Y, OFFSET_X, OFFSET_Y
    if not os.path.exists(CONFIG_FILE):
        print("⚠️ 未找到配置文件，使用默认值")
        return
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        CONF_THRESHOLD = data.get('CONF_THRESHOLD', CONF_THRESHOLD)
        FOOT_OFFSET_Y = data.get('FOOT_OFFSET_Y', FOOT_OFFSET_Y)
        OFFSET_X = data.get('offset_x', OFFSET_X)
        OFFSET_Y = data.get('offset_y', OFFSET_Y)
        print(f"✅ 加载配置: 阈值={CONF_THRESHOLD:.2f}, FOOT_OFFSET_Y={FOOT_OFFSET_Y}, offset_x={OFFSET_X}, offset_y={OFFSET_Y}")
    except Exception as e:
        print(f"⚠️ 加载配置失败: {e}，使用默认值")

def save_config(overlay=None):
    offset_x = overlay.offset_x if overlay else OFFSET_X
    offset_y = overlay.offset_y if overlay else OFFSET_Y
    data = {
        'CONF_THRESHOLD': CONF_THRESHOLD,
        'FOOT_OFFSET_Y': FOOT_OFFSET_Y,
        'offset_x': offset_x,
        'offset_y': offset_y
    }
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"✅ 配置已保存: {CONFIG_FILE}")
    except Exception as e:
        print(f"⚠️ 保存配置失败: {e}")

# ---------- 截图保存函数 ----------
def save_screenshot():
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            save_path = os.path.join(CONFIG_DIR, "image.png")
            cv2.imwrite(save_path, img)
            print(f"✅ 截图已保存: {save_path}")
    except Exception as e:
        print(f"⚠️ 截图保存失败: {e}")

# ---------- 标定数据 ----------
screen_points = np.array([
    [136.5, 384.5],
    [2380.5 + 12, 384.5],
    [2520.5 + 12, 1264.5 - 10],
    [11.5, 1264.5 - 10]
], dtype=np.float32)
world_points = np.array([
    [-118,  86],
    [ 118,  86],
    [ 118, -86],
    [-118, -86]
], dtype=np.float32)

M_screen_to_world = cv2.getPerspectiveTransform(screen_points, world_points)
M_world_to_screen = cv2.getPerspectiveTransform(world_points, screen_points)

def screen_to_world(px, py):
    pts = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pts, M_screen_to_world)
    wx, wz = result[0][0][0], result[0][0][1]
    wx = max(-118.0, min(118.0, wx))
    wz = max(-86.0, min(86.0, wz))
    return wx, wz

def world_to_screen(wx, wz):
    pts = np.array([[[wx, wz]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pts, M_world_to_screen)
    return result[0][0][0], result[0][0][1]

# ---------- 闪现落点计算 ----------
def compute_flash_target(A, E):
    Ax, Az = A
    Ex, Ez = E
    dx = Ex - Ax
    dz = Ez - Az
    length = math.hypot(dx, dz)
    if length < 1e-6:
        return None
    dir_x = dx / length
    dir_z = dz / length

    d_max = 118.0
    if dir_x > 0:
        d_max = min(d_max, (0 - Ax) / dir_x)
    if dir_x < 0:
        d_max = min(d_max, (-118 - Ax) / dir_x)
    if dir_z > 0:
        d_max = min(d_max, (86 - Az) / dir_z)
    elif dir_z < 0:
        d_max = min(d_max, (-86 - Az) / dir_z)
    if d_max <= 0:
        return None
    d = d_max
    Bx = Ax + d * dir_x
    Bz = Az + d * dir_z
    if Bx > 0:
        Bx = 0
    if Bx < -118:
        Bx = -118
    return (Bx, Bz)

# ---------- 信息板 ----------
class InfoBoard(QWidget):
    visibilityChanged = pyqtSignal(bool)
    thresholdChanged = pyqtSignal(float)   # 用于通知外部更新

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedSize(300, 90)          # 增加高度显示阈值
        screen = QApplication.primaryScreen()
        size = screen.size()
        self.move(size.width() - 320, size.height() - 110)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: rgba(0,0,0,150); color: white; border-radius: 8px; padding: 8px;")
        self.label.setFont(QFont("Consolas", 11))
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.fps = 0
        self.draw_enabled = True
        self._is_visible = True
        self.threshold = CONF_THRESHOLD
        self.update_text()

    def update_text(self):
        status = "ON" if self.draw_enabled else "OFF"
        color = "#66ff66" if self.draw_enabled else "#ffaa00"
        text = f"FPS: {self.fps}  阈值: {self.threshold:.2f}\n绘制: {status}"
        self.label.setText(text)
        self.label.setStyleSheet(
            f"background-color: rgba(0,0,0,180); color: {color}; border-radius: 8px; padding: 8px;"
        )

    @pyqtSlot(int)
    def update_fps(self, fps):
        self.fps = fps
        self.update_text()

    @pyqtSlot(bool)
    def update_draw(self, enabled):
        self.draw_enabled = enabled
        self.update_text()

    @pyqtSlot(float)
    def update_threshold(self, th):
        self.threshold = th
        self.update_text()

    @pyqtSlot()
    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
            self._is_visible = False
        else:
            self.show()
            self._is_visible = True
        self.visibilityChanged.emit(self._is_visible)

    def is_visible(self):
        return self._is_visible

# ---------- 覆盖层窗口 ----------
class DetectionOverlay(QWidget):
    drawStateChanged = pyqtSignal(bool)

    def __init__(self, initial_offset_x=0, initial_offset_y=0):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        screen = QApplication.primaryScreen()
        size = screen.size()
        self.setGeometry(0, 0, size.width(), size.height())
        self.show()

        self.detections = []
        self.draw_enabled = True
        self.offset_x = initial_offset_x
        self.offset_y = initial_offset_y

    @pyqtSlot(list)
    def set_detections(self, dets):
        self.detections = dets
        self.update()

    @pyqtSlot()
    def toggle_draw(self):
        self.draw_enabled = not self.draw_enabled
        self.drawStateChanged.emit(self.draw_enabled)
        self.update()

    @pyqtSlot(int, int)
    def move_offset(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy
        self.update()

    def paintEvent(self, event):
        if not self.draw_enabled:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for (x1, y1, x2, y2, label) in self.detections:
            # 绘制黑色边框，线宽改为 2（原为 4）
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            if label == 'mb':
                color = QColor(0, 255, 0)
            else:
                color = QColor(255, 0, 0)
            painter.setPen(QPen(color, 1))
            painter.drawText(x1, y1 - 5, f"{label}")

            foot_x = (x1 + x2) // 2 + self.offset_x
            foot_y = y2 + FOOT_OFFSET_Y + self.offset_y
            painter.setBrush(QColor(255, 255, 0))
            painter.setPen(QPen(QColor(255, 255, 0), 1))
            painter.drawEllipse(foot_x - 5, foot_y - 5, 10, 10)

# ---------- 检测工作线程 ----------
class DetectionWorker(QThread):
    detections_signal = pyqtSignal(list)
    fps_signal = pyqtSignal(int)
    threshold_signal = pyqtSignal(float)

    def __init__(self, info_board):
        super().__init__()
        self.info_board = info_board
        self.running = True
        self.visible = True
        self.model = None
        self.sct = mss.MSS()
        self.current_threshold = CONF_THRESHOLD

        # ---------- 抽帧优化：推理间隔（每 N 帧推理一次） ----------
        self.inference_interval = 2          # 可手动修改此值（例如 3 或 4）
        self.last_dets = []                  # 缓存上一次检测结果

        monitor_info = self.sct.monitors[1]
        screen_w = monitor_info["width"]
        screen_h = monitor_info["height"]

        pts = np.array([
            [136.5, 384.5],
            [2380.5 + 12, 384.5],
            [2520.5 + 12, 1264.5 - 10],
            [11.5, 1264.5 - 10]
        ], dtype=np.float32)

        min_x_raw = int(np.min(pts[:, 0])) - 50
        max_x_raw = int(np.max(pts[:, 0])) + 50
        min_y_raw = 0
        max_y_raw = min(screen_h, int(np.max(pts[:, 1])) + 100)

        self.crop_offset_x = max(0, min_x_raw)
        self.crop_offset_y = max(0, min_y_raw)
        crop_right = min(screen_w, max_x_raw)
        crop_bottom = min(screen_h, max_y_raw)

        if crop_right - self.crop_offset_x < 100 or crop_bottom - self.crop_offset_y < 100:
            print("⚠️ 裁剪区域过小，使用全屏")
            self.monitor = self.sct.monitors[1]
            self.crop_offset_x = 0
            self.crop_offset_y = 0
        else:
            self.monitor = {
                "left": self.crop_offset_x,
                "top": self.crop_offset_y,
                "width": crop_right - self.crop_offset_x,
                "height": crop_bottom - self.crop_offset_y
            }
            print(f"✅ 截图区域: {self.monitor}")

        self.screen_width = screen_w
        self.screen_height = screen_h

        self.my_pos_world = None
        self.enemy_pos_world = None
        self.busy = False

        self.info_board.visibilityChanged.connect(self.set_visible)

    @pyqtSlot(bool)
    def set_visible(self, visible):
        self.visible = visible

    @pyqtSlot(float)
    def set_threshold(self, th):
        """外部调节阈值，同时通知信息板"""
        self.current_threshold = max(0.05, min(0.95, th))
        global CONF_THRESHOLD
        CONF_THRESHOLD = self.current_threshold
        self.threshold_signal.emit(self.current_threshold)
        print(f"🔧 置信度阈值调整为: {self.current_threshold:.2f}")

    def run(self):
        print("加载 ONNX 模型 (Ultralytics)...")
        self.model = YOLO(MODEL_PATH)
        if torch.cuda.is_available():
            print("✅ GPU 可用 (ONNX Runtime 将使用 CUDA)")
        else:
            print("⚠️ CPU 模式")

        print(f"开始实时检测（抽帧优化：每 {self.inference_interval} 帧推理一次）")

        frame_counter = 0
        fps_counter = 0
        fps_timer = time.time()

        while self.running:
            if not self.visible:
                time.sleep(0.1)
                continue

            frame_counter += 1
            fps_counter += 1

            # ---------- 跳帧判断 ----------
            if frame_counter % self.inference_interval != 0:
                # 跳过本帧推理，复用上一次检测结果和坐标
                self.detections_signal.emit(self.last_dets)
            else:
                # 执行截图与推理
                try:
                    screenshot = self.sct.grab(self.monitor)
                    img = np.array(screenshot)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                except Exception as e:
                    print(f"截图失败: {e}")
                    time.sleep(0.1)
                    continue

                try:
                    results = self.model(img, conf=self.current_threshold, iou=0.5, imgsz=800, verbose=False)
                except Exception as e:
                    print(f"推理失败: {e}")
                    time.sleep(0.1)
                    continue

                dets = []
                my_foot = None
                enemy_foot = None

                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            cls_id = int(box.cls[0])
                            label = CLASS_NAMES.get(cls_id, "unknown")

                            abs_x1 = x1 + self.crop_offset_x
                            abs_y1 = y1 + self.crop_offset_y
                            abs_x2 = x2 + self.crop_offset_x
                            abs_y2 = y2 + self.crop_offset_y

                            foot_x = (abs_x1 + abs_x2) // 2
                            foot_y = abs_y2 + FOOT_OFFSET_Y

                            dets.append((abs_x1, abs_y1, abs_x2, abs_y2, label))

                            if 0 <= foot_x < self.screen_width and 0 <= foot_y < self.screen_height:
                                if label == 'mb' and my_foot is None:
                                    my_foot = (foot_x, foot_y)
                                elif label == 'eb' and enemy_foot is None:
                                    enemy_foot = (foot_x, foot_y)

                # 更新世界坐标
                if my_foot is not None:
                    wx, wz = screen_to_world(my_foot[0], my_foot[1])
                    self.my_pos_world = (wx, wz)
                else:
                    self.my_pos_world = None

                if enemy_foot is not None:
                    wx, wz = screen_to_world(enemy_foot[0], enemy_foot[1])
                    self.enemy_pos_world = (wx, wz)
                else:
                    self.enemy_pos_world = None

                # 缓存结果供跳帧使用
                self.last_dets = dets
                self.detections_signal.emit(dets)

            # FPS 统计（不区分是否推理，保持帧率计数）
            now = time.time()
            if now - fps_timer >= 1.0:
                self.fps_signal.emit(fps_counter)
                fps_counter = 0
                fps_timer = now

            time.sleep(0.005)

        print("检测线程结束")

    @pyqtSlot()
    def flash(self):
        if self.busy:
            print("操作进行中")
            return
        if self.my_pos_world is None or self.enemy_pos_world is None:
            print("位置未就绪")
            return

        self.busy = True
        try:
            B = compute_flash_target(self.my_pos_world, self.enemy_pos_world)
            if B is None:
                print("落点无效")
                self.busy = False
                return
            bx, by = world_to_screen(B[0], B[1])
            bx, by = int(bx), int(by)
            bx = max(0, min(self.screen_width - 1, bx))
            by = max(0, min(self.screen_height - 1, by))
            print(f"闪现落点: 世界({B[0]:.1f}, {B[1]:.1f}) -> 屏幕({bx}, {by})")
            pyautogui.click(bx, by)
            print("✅ 已点击闪现")
        except Exception as e:
            print(f"闪现异常: {e}")
        finally:
            self.busy = False

    def stop(self):
        self.running = False

# ---------- 主程序 ----------
def main():
    load_config()

    app = QApplication(sys.argv)

    overlay = DetectionOverlay(initial_offset_x=OFFSET_X, initial_offset_y=OFFSET_Y)
    info_board = InfoBoard()
    info_board.show()

    info_board.update_fps(0)
    info_board.update_draw(overlay.draw_enabled)
    info_board.update_threshold(CONF_THRESHOLD)

    overlay.drawStateChanged.connect(info_board.update_draw)
    worker = DetectionWorker(info_board)
    worker.detections_signal.connect(overlay.set_detections)
    worker.fps_signal.connect(info_board.update_fps)
    worker.threshold_signal.connect(info_board.update_threshold)  # 显示阈值变化
    worker.start()

    # ---------- 热键回调 ----------
    def flash_callback():
        if not info_board.is_visible():
            return
        QMetaObject.invokeMethod(worker, "flash", Qt.QueuedConnection)

    def toggle_draw_callback():
        QMetaObject.invokeMethod(overlay, "toggle_draw", Qt.QueuedConnection)

    def toggle_info_callback():
        QMetaObject.invokeMethod(info_board, "toggle_visibility", Qt.QueuedConnection)

    def exit_callback():
        save_config(overlay)
        worker.stop()
        QMetaObject.invokeMethod(app, "quit", Qt.QueuedConnection)

    def save_callback():
        save_config(overlay)

    def screenshot_callback():
        save_screenshot()

    # 阈值调节
    def threshold_up():
        new_th = CONF_THRESHOLD + 0.05
        QMetaObject.invokeMethod(worker, "set_threshold", Qt.QueuedConnection, Q_ARG(float, new_th))

    def threshold_down():
        new_th = CONF_THRESHOLD - 0.05
        QMetaObject.invokeMethod(worker, "set_threshold", Qt.QueuedConnection, Q_ARG(float, new_th))

    STEP = 5
    def move_offset_callback(dx, dy):
        QMetaObject.invokeMethod(overlay, "move_offset", Qt.QueuedConnection,
                                 Q_ARG(int, dx), Q_ARG(int, dy))

    keyboard.add_hotkey('f', flash_callback)
    keyboard.add_hotkey('r', toggle_draw_callback)
    keyboard.add_hotkey('f5', toggle_info_callback)
    keyboard.add_hotkey('esc', exit_callback)
    keyboard.add_hotkey('ctrl+s', save_callback)
    keyboard.add_hotkey('ctrl+p', screenshot_callback)
    keyboard.add_hotkey('[', threshold_down)   # 降低阈值
    keyboard.add_hotkey(']', threshold_up)     # 提高阈值

    keyboard.add_hotkey('w', lambda: move_offset_callback(0, -STEP))
    keyboard.add_hotkey('a', lambda: move_offset_callback(-STEP, 0))
    keyboard.add_hotkey('s', lambda: move_offset_callback(0, STEP))
    keyboard.add_hotkey('d', lambda: move_offset_callback(STEP, 0))

    print("✅ 热键: R切换绘制, F5切换信息板, F闪现, ESC退出, Ctrl+S保存配置, Ctrl+P截图")
    print("   [ 降低阈值, ] 提高阈值 (每次 ±0.05)")
    print(f"✅ WASD 微调圆点位置（每按一次移动 {STEP} 像素）")
    print("✅ 配置自动保存于退出时，也可手动按 Ctrl+S 保存")
    print(f"📁 配置文件: {CONFIG_FILE}")
    print("📸 按 Ctrl+P 保存当前屏幕截图到 ./set/image.png")

    exit_code = app.exec_()
    worker.stop()
    worker.wait()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()