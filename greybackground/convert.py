from pathlib import Path
from PIL import Image

def convert_png_to_jpg(folder_path):
    folder = Path(folder_path)
    for png_path in folder.rglob("*.png"):
        img = Image.open(png_path).convert("RGB")
        jpg_path = png_path.with_suffix(".jpg")
        img.save(jpg_path, "JPEG", quality=95)
        png_path.unlink()  # 删除原png文件
        # print(f"转换: {png_path} -> {jpg_path}")

if __name__ == "__main__":
    print("正在转换PNG文件为JPG...")
    convert_png_to_jpg("./true-image-eeg-data")  # 修改为你的文件夹路径