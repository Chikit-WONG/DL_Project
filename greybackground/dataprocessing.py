import os
from PIL import Image
import glob

def create_composite_with_image_b(image_a_path, image_b, output_path, bg_size=(1280, 1280), bg_color=(158, 162, 166)):
    """
    将图片 A 处理后与图片 B 合成
    """
    # 创建背景
    background = Image.new("RGB", bg_size, bg_color)
    
    # 处理图片 A
    img_a = Image.open(image_a_path).convert("RGB")
    min_side = min(img_a.width, img_a.height)
    left = (img_a.width - min_side) // 2
    top = (img_a.height - min_side) // 2
    img_a_square = img_a.crop((left, top, left + min_side, top + min_side))
    img_a_resized = img_a_square.resize((640, 640), Image.LANCZOS)
    
    # 计算中央位置
    center_x = bg_size[0] // 2
    center_y = bg_size[1] // 2
    
    # 粘贴图片 A
    a_left = center_x - img_a_resized.width // 2
    a_top = center_y - img_a_resized.height // 2
    background.paste(img_a_resized, (a_left, a_top))
    
    # 粘贴图片 B（保持透明）
    b_left = center_x - image_b.width // 2
    b_top = center_y - image_b.height // 2
    background.paste(image_b, (b_left, b_top), mask=image_b)
    
    # 保存
    background.save(output_path)

def process_all_images(image_b_path, input_root, output_root):
    """
    遍历所有图片 A，与固定的图片 B 合成
    """
    # 加载图片 B（只需加载一次）
    print(f"加载图片 B: {image_b_path}")
    image_b = Image.open(image_b_path).convert("RGBA")
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    
    # 递归查找所有图片
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(input_root, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
        pattern_upper = os.path.join(input_root, '**', ext.upper())
        image_files.extend(glob.glob(pattern_upper, recursive=True))
    
    image_files = sorted(list(set(image_files)))
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每个图片
    for i, img_path in enumerate(image_files, 1):
        try:
            # 计算输出路径
            rel_path = os.path.relpath(img_path, input_root)
            output_path = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_path = os.path.splitext(output_path)[0] + '.png'
            
            # print(f"[{i}/{len(image_files)}] 处理: {rel_path}")
            
            # 合成图片
            create_composite_with_image_b(img_path, image_b, output_path)
            
        except Exception as e:
            print(f"  错误: {img_path} - {str(e)}")
    
    print(f"\n完成！所有图片已输出到: {output_root}")

if __name__ == "__main__":
    # 配置
    input_folder = "./image-eeg-data"
    output_folder = "./true-image-eeg-data"
    image_b_path = "redpoint.png"  # 请修改为你的图片 B 路径
    
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在！")
    elif not os.path.exists(image_b_path):
        print(f"错误: 图片 B 文件 '{image_b_path}' 不存在！")
    else:
        process_all_images(image_b_path, input_folder, output_folder)