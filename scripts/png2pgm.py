import os
from PIL import Image

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATASET'), 'r') as f:
    DATASET_NAME = f.read().strip()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

INPUT_PNG_PATH = os.path.join(DATASET_ROOT, 'global_map_edited.png')
OUTPUT_PGM_PATH = os.path.join(DATASET_ROOT, 'global_map_edited.pgm')

def main():
    if not os.path.exists(INPUT_PNG_PATH):
        print(f"Error: Input file {INPUT_PNG_PATH} does not exist.")
        return

    print(f"Loading {INPUT_PNG_PATH}...")
    # 使用 PIL 打开图像并转换为灰度图 (L 模式)
    img = Image.open(INPUT_PNG_PATH).convert('L')
    
    # 保存为 PGM 格式
    # 保存时 PIL 会根据扩展名自动选择格式
    img.save(OUTPUT_PGM_PATH)
    
    print(f"Successfully converted to: {OUTPUT_PGM_PATH}")
    print(f"Image size: {img.size}")

if __name__ == "__main__":
    main()
