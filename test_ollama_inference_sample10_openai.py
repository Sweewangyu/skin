import os
import glob
import base64
import pandas as pd
from openai import OpenAI
import time

# --- 1. 配置 ---

# 配置OpenAI客户端以连接到本地Ollama服务器
client = OpenAI(
    api_key="ollama",  # 对于本地Ollama，API密钥可以是占位符
    base_url="http://localhost:11434/v1",
)

# 数据集路径（仅使用第一个分块）
IMAGE_DIR_1 = "data/HAM10000/HAM10000_images_part_1"
METADATA_FILE = "data/HAM10000/HAM10000_metadata.csv"

# 用于分类的类别列表
categories = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
categories_str = ", ".join(categories)

# 视觉语言模型的提示
PROMPT_TEXT = f"""
你是一名世界级的皮肤科AI诊断助手。请按照以下步骤分析和分类提供的皮肤镜图像。

**第一步：特征分析（内心思考，不要输出）**
1.  **对称性**：病变的形状是否对称？
2.  **边缘**：边缘是清晰规则，还是模糊、不规则、有切迹？
3.  **颜色**：颜色是单一均匀，还是包含多种颜色（如棕、黑、红、蓝、白）？颜色分布是否均匀？
4.  **结构**：是否能观察到特定的皮肤镜结构？例如，色素网络、点状血管、树枝状血管、蓝白幕、乳头状结构等。
5.  **整体评估**：综合以上特征，病变给人的整体感觉是良性的（有序、规则）还是恶性的（混乱、不规则）？

**第二步：分类判断**
根据你的分析，将图像归类到以下七个类别之一：{categories_str}。
    - 黑色素瘤 (mel, Melanoma)
      特征：明显的不对称性、不规则边缘、颜色多样性（棕、黑、蓝、白、红等），  
      非典型色素网络、蓝白幕、放射状线条、负网状结构、不对称的小点或条纹、局部回避区等恶性特征。 
 
    - 基底细胞癌 (bcc, Basal Cell Carcinoma)
      特征：树枝状血管、蓝灰色卵圆巢、光滑珠光边缘、溃疡或结痂区域、车轮辐射状结构、白色条纹或亮点。 
 
    - 黑色素细胞痣 (nv, Melanocytic Nevus) 
      特征：整体对称、规则的色素网络、均匀的棕色色调、清晰边界、  
      可见规则点状或球状结构、均匀分布的色素网格。 
 
    - 脂溢性角化病 (bkl, Benign Keratosis)**  
      特征：粉刺样开口、脑回状（丘脑状）结构、粘贴感外观、白色假网状结构、角质栓、黑点或伪毛囊口。 
 
    - 光化性角化病 (akiec, Actinic Keratosis)
      特征：红白交错的表面、毛细血管扩张、鳞屑、角质过度增生、淡棕或红色调，  
      可能可见“草地样”或“红白斑块状”结构。 
 
    - 皮肤纤维瘤 (df, Dermatofibroma)
      特征：中心棕色区伴周围淡色晕、放射状色素结构、中心瘢痕样白区、周边色素网络逐渐消退、轻微凹陷。 
 
    - 血管性病变 (vasc, Vascular Lesion)
      特征：均匀的红色至紫色区域、清晰可见的血管结构、点状或线状血管、湖状血管样分布、整体对称。

**第三步：输出结果**
请只输出最终确定的类别缩写。不要包含任何分析、解释或额外文字。
"""

# --- 2. 辅助函数 ---

def encode_image(image_path):
    """将图像文件编码为base64字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"编码图像 {image_path} 时出错: {e}")
        return None

def classify_image_timed(image_id, image_path):
    """
    使用Ollama模型对单个图像进行分类，并测量调用耗时。
    返回 (predicted_class, duration_seconds)。
    """
    base64_image = encode_image(image_path)
    if not base64_image:
        return "error", 0.0

    try:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model="qwen2.5vl:32b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_TEXT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=8,  # 期望得到一个简短的类别名称
        )
        duration = time.perf_counter() - start

        predicted_class = response.choices[0].message.content.strip().lower()

        # 验证模型的输出
        if predicted_class not in categories:
            print(f"警告：图像 {image_id} 的模型返回了一个意外的类别 '{predicted_class}'。将其设置为'unknown'。")
            predicted_class = "unknown"

        return predicted_class, duration

    except Exception as e:
        print(f"分类图像 {image_id} 时出错: {e}")
        return "error", 0.0

# --- 3. 主流程（仅选取10张） ---

def main():
    print("--- 开始 10 张图像测试 ---")

    # 加载元数据
    metadata_df = pd.read_csv(METADATA_FILE)
    print(f"已加载 {len(metadata_df)} 张图像的元数据。")

    # 仅第一个分块的图像路径
    image_paths_part1 = glob.glob(os.path.join(IMAGE_DIR_1, '*.jpg'))
    image_id_to_path = {os.path.basename(p).split('.')[0]: p for p in image_paths_part1}
    print(f"在第一个分块中共找到 {len(image_paths_part1)} 张图像。")

    # 从元数据中选取在 part_1 中存在的前10张
    selected_ids = []
    for img_id in metadata_df['image_id']:
        if img_id in image_id_to_path:
            selected_ids.append(img_id)
        if len(selected_ids) >= 10:
            break

    if not selected_ids:
        print("未在第一个分块中找到可用图像。")
        return

    print(f"将测试的图像数量：{len(selected_ids)}（最多10张）。")
    durations = []
    predictions = {}

    # 顺序测试，逐次打印调用耗时
    for idx, image_id in enumerate(selected_ids, start=1):
        image_path = image_id_to_path[image_id]
        predicted_class, duration = classify_image_timed(image_id, image_path)
        durations.append(duration)
        predictions[image_id] = predicted_class
        print(f"[{idx:02d}/{len(selected_ids)}] 图像 {image_id}: 预测='{predicted_class}', 调用耗时={duration:.3f} 秒")

    # 平均耗时
    avg_time = sum(durations) / len(durations) if durations else 0.0
    print(f"\n平均调用耗时：{avg_time:.3f} 秒")

    # --- 4. 评估准确率（仅针对选取的10张） ---
    print("\n--- 正在评估准确率（10张） ---")
    # 构建结果 DataFrame
    results_list = [{"image_id": image_id, "predicted_class": predictions[image_id]} for image_id in selected_ids]
    results_df = pd.DataFrame(results_list)

    # 真实标签
    gt_df = metadata_df[metadata_df['image_id'].isin(selected_ids)][['image_id', 'dx']].rename(columns={'dx': 'true_class'})

    merged_df = pd.merge(results_df, gt_df, on='image_id', how='inner')
    total_predictions = len(merged_df)
    correct_predictions = (merged_df['predicted_class'] == merged_df['true_class']).sum()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    print(f"总评估预测数: {total_predictions}")
    print(f"正确预测数: {correct_predictions}")
    print(f"总体准确率: {accuracy:.4f}")

if __name__ == "__main__":
    main()