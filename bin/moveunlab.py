import os
import shutil
from pathlib import Path

# ========== 配置（请根据实际情况调整） ==========
BASE_DIR = Path("D:/MAG/b")

# 源目录（当前训练/验证图片所在位置）
TRAIN_IMG_SRC = BASE_DIR / "dataset" / "images" / "train"
VAL_IMG_SRC   = BASE_DIR / "dataset" / "images" / "val"

# 对应的标注目录（用于判断是否已标注）
TRAIN_LABEL_SRC = BASE_DIR / "dataset" / "labels" / "train"
VAL_LABEL_SRC   = BASE_DIR / "dataset" / "labels" / "val"

# 目标目录（存放未标注的图片）
TRAIN_IMG_DST = BASE_DIR / "predataset" / "images" / "train"
VAL_IMG_DST   = BASE_DIR / "predataset" / "images" / "val"
# ================================================

def move_unlabeled_images(src_img_dir, src_label_dir, dst_img_dir):
    """
    将 src_img_dir 中没有对应标注的图片移动到 dst_img_dir
    """
    # 确保目标目录存在
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件（支持常见格式）
    image_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in src_img_dir.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]
    if not images:
        print(f"⚠️ 在 {src_img_dir} 中没有找到图片文件。")
        return 0

    # 获取所有标注文件名（不含扩展名）
    label_stems = {f.stem for f in src_label_dir.glob("*.txt")}

    moved_count = 0
    for img in images:
        if img.stem not in label_stems:
            # 未标注，移动
            try:
                shutil.move(str(img), str(dst_img_dir / img.name))
                moved_count += 1
                print(f"📦 移动未标注图片: {img.name} -> {dst_img_dir}")
            except Exception as e:
                print(f"❌ 移动失败 {img.name}: {e}")
    return moved_count

def main():
    print("=" * 60)
    print("🚀 未标注图片清理工具")
    print("=" * 60)

    # 处理训练集
    print("\n🔍 正在检查训练集...")
    moved_train = move_unlabeled_images(TRAIN_IMG_SRC, TRAIN_LABEL_SRC, TRAIN_IMG_DST)
    print(f"✅ 训练集共移走 {moved_train} 张未标注图片。")

    # 处理验证集
    print("\n🔍 正在检查验证集...")
    moved_val = move_unlabeled_images(VAL_IMG_SRC, VAL_LABEL_SRC, VAL_IMG_DST)
    print(f"✅ 验证集共移走 {moved_val} 张未标注图片。")

    # 统计剩余已标注图片数量
    remaining_train = len([f for f in TRAIN_IMG_SRC.glob("*") if f.suffix.lower() in ('.png','.jpg','.jpeg')])
    remaining_val = len([f for f in VAL_IMG_SRC.glob("*") if f.suffix.lower() in ('.png','.jpg','.jpeg')])

    print("\n" + "=" * 60)
    print("📊 清理完成后的统计")
    print("=" * 60)


    print(f"训练集剩余已标注图片: {remaining_train} 张")
    print(f"验证集剩余已标注图片: {remaining_val} 张")
    print(f"未标注图片已移至: {BASE_DIR / 'predataset' / 'images'}")
    print("=" * 60)

    if remaining_train == 0 and remaining_val == 0:
        print("⚠️ 当前没有任何已标注的图片！请先用 LabelImg 标注后再运行本脚本。")
    else:
        print("✅ 现在可以直接开始训练了（使用 dataset 目录下的图片）。")

if __name__ == "__main__":
    main()