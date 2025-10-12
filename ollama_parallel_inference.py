# 顶部配置区域（更新 categories 与 PROMPT_TEXT）
import os
import glob
import base64
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
from PIL import Image

# --- 1. 配置 ---

# 配置OpenAI客户端以连接到本地Ollama服务器
client = OpenAI(
    api_key="ollama",  # 对于本地Ollama，API密钥可以是占位符
    base_url="http://localhost:11434/v1",
)

# 数据集路径
IMAGE_DIR = "ISIC2019/ISIC_2019_Test_Input"
METADATA_FILE = "ISIC2019/ISIC_2019_Test_GroundTruth.csv"

# 结果和检查点路径
RESULTS_FILE = "isic2019_ollama_classification_results.csv"
CHECKPOINT_FILE = "isic2019_ollama_checkpoint.json"

# 用于分类的类别列表 (ISIC2019有8个类别，不包括UNK)
categories = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
categories_str = ", ".join(categories)

# 视觉语言模型的提示（加入 UNK，并在不确定时输出 UNK）
PROMPT_TEXT = f"""
你是一名世界级的皮肤科AI诊断助手。请按照以下步骤分析和分类提供的皮肤镜图像。

**第一步：特征分析（内心思考，不要输出）**
1.  **对称性**：病变的形状是否对称？
2.  **边缘**：边缘是清晰规则，还是模糊、不规则、有切迹？
3.  **颜色**：颜色是单一均匀，还是包含多种颜色（如棕、黑、红、蓝、白）？颜色分布是否均匀？
4.  **结构**：是否能观察到特定的皮肤镜结构？例如，色素网络、点状血管、树枝状血管、蓝白幕、乳头状结构等。
5.  **整体评估**：综合以上特征，病变给人的整体感觉是良性的（有序、规则）还是恶性的（混乱、不规则）？

**第二步：分类判断**
根据你的分析，将图像归类到以下九个类别之一：{categories_str}。
    - 黑色素瘤 (MEL)
      特征：明显的不对称性、不规则边缘、颜色多样性（棕、黑、蓝、白、红等），  
      非典型色素网络、蓝白幕、放射状线条、负网状结构、不对称的小点或条纹、局部回避区等恶性特征。 
 
    - 黑色素细胞痣 (NV) 
      特征：整体对称、规则的色素网络、均匀的棕色色调、清晰边界、  
      可见规则点状或球状结构、均匀分布的色素网格。 
 
    - 基底细胞癌 (BCC)
      特征：树枝状血管、蓝灰色卵圆巢、光滑珠光边缘、溃疡或结痂区域、车轮辐射状结构、白色条纹或亮点。 
 
    - 光化性角化病 (AK)
      特征：红白交错的表面、毛细血管扩张、鳞屑、角质过度增生、淡棕或红色调，  
      可能可见"草地样"或"红白斑块状"结构。 
 
    - 脂溢性角化病 (BKL)  
      特征：粉刺样开口、脑回状（丘脑状）结构、粘贴感外观、白色假网状结构、角质栓、黑点或伪毛囊口。 
 
    - 皮肤纤维瘤 (DF)
      特征：中心棕色区伴周围淡色晕、放射状色素结构、中心瘢痕样白区、周边色素网络逐渐消退、轻微凹陷。 
 
    - 血管性病变 (VASC)
      特征：均匀的红色至紫色区域、清晰可见的血管结构、点状或线状血管、湖状血管样分布、整体对称。
      
    - 鳞状细胞癌 (SCC)
      特征：鳞屑、结痂、角化、溃疡、不规则血管、白色或黄色角质区域、边界不清晰。

    - 未知类别 (UNK)
      特征：当图像特征不足以可靠分类为上述任何一类、或不属于上述类别时，应输出 UNK。

**第三步：输出结果**
请只输出最终确定的类别缩写（MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK）。如果不确定，请输出 UNK。不要包含任何分析、解释或额外文字。
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

# classify_image 函数（将不在类别列表中的预测回退为 UNK）
def classify_image(image_id, image_path):
    """
    使用Ollama模型对单个图像进行分类。
    返回一个元组 (image_id, result_dict)。
    """
    base64_image = encode_image(image_path)
    if not base64_image:
        return image_id, None

    try:
        response = client.chat.completions.create(
            model="minicpm-v:8b",
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
            max_tokens=10, # 期望得到一个简短的类别名称
        )
        predicted_class = response.choices[0].message.content.strip().upper()

        # 验证模型的输出
        if predicted_class not in categories:
            print(f"警告：图像 {image_id} 的模型返回了一个意外的类别 '{predicted_class}'。将其设置为'UNK'。")
            predicted_class = "UNK"

        return image_id, {"predicted_class": predicted_class}

    except Exception as e:
        print(f"分类图像 {image_id} 时出错: {e}")
        return image_id, None

def load_checkpoint():
    """从检查点文件加载已处理的图像和结果。"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get("processed_images", [])), data.get("results", {})
    return set(), {}

def save_checkpoint(processed_images, results):
    """将已处理的图像和结果保存到检查点文件。"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"processed_images": list(processed_images), "results": results}, f, indent=4)

# --- 3. 主要执行流程 ---

# main 函数（不再过滤 UNK；评估包含 UNK）
def main():
    """运行分类过程的主函数。"""
    print("--- 开始图像分类 ---")

    # 加载元数据
    metadata_df = pd.read_csv(METADATA_FILE)
    print(f"已加载 {len(metadata_df)} 张图像的元数据。")
    
    # 不再过滤 UNK，直接使用所有元数据
    # 过去版本中这里会过滤 UNK 的样本，现在改为全量评估
    # 获取所有图像路径
    image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    image_id_to_path = {os.path.basename(p).split('.')[0]: p for p in image_paths}
    print(f"共找到 {len(image_paths)} 张图像。")

    # 从检查点加载
    processed_ids, results = load_checkpoint()
    print(f"已加载检查点。{len(processed_ids)} 张图像已被处理。")

    # 确定要处理的图像（所有类别，包括 UNK）
    unprocessed_tasks = []
    for img_id in metadata_df['image']:
        if img_id not in processed_ids and img_id in image_id_to_path:
            unprocessed_tasks.append((img_id, image_id_to_path[img_id]))
    
    print(f"找到 {len(unprocessed_tasks)} 张待处理的图像。")

    if not unprocessed_tasks:
        print("没有新的图像需要处理。")
    else:
        # 并行处理
        with ProcessPoolExecutor() as executor:
            # 为未处理的图像创建future
            futures = {executor.submit(classify_image, img_id, path): (img_id, path) for img_id, path in unprocessed_tasks}
            
            # 在任务完成时处理结果
            for future in tqdm(as_completed(futures), total=len(unprocessed_tasks), desc="正在分类图像"):
                image_id, result = future.result()
                if result:
                    results[image_id] = result
                    processed_ids.add(image_id)
                    
                    # 定期保存检查点（例如，每处理10张图像）
                    if len(processed_ids) % 10 == 0:
                        save_checkpoint(processed_ids, results)

    # 最后保存检查点和结果
    print("分类完成。正在保存最终结果...")
    save_checkpoint(processed_ids, results)

    # 将结果转换为DataFrame并保存为CSV
    results_list = []
    for image_id, result_data in results.items():
        results_list.append({
            "image_id": image_id,
            "predicted_class": result_data.get("predicted_class", "UNK")
        })
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"结果已保存到 {RESULTS_FILE}")

    # --- 4. 评估 ---
    print("\n--- 正在评估准确率 ---")
    
    # 获取真实标签（ISIC2019使用one-hot编码，需要转换）
    gt_subset = metadata_df[metadata_df['image'].isin(results_df['image_id'])]
    
    # 将one-hot编码转换为类别标签（包含 UNK）
    true_classes = []
    for _, row in gt_subset.iterrows():
        image_id = row['image']
        # 找出值为1.0的列名（类别）
        found = False
        for cat in categories:
            if row.get(cat, 0.0) == 1.0:
                true_classes.append({"image_id": image_id, "true_class": cat})
                found = True
                break
        # 如果没有任何类别为 1.0（极少数异常），标记为 UNK
        if not found:
            true_classes.append({"image_id": image_id, "true_class": "UNK"})
    
    gt_df = pd.DataFrame(true_classes)
    
    # 合并预测结果和真实标签
    merged_df = pd.merge(results_df, gt_df, left_on='image_id', right_on='image_id', how='inner')

    correct_predictions = (merged_df['predicted_class'] == merged_df['true_class']).sum()
    total_predictions = len(merged_df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"总评估预测数: {total_predictions}")
    print(f"正确预测数: {correct_predictions}")
    print(f"总体准确率: {accuracy:.4f}")

    # 各类别准确率
    class_accuracy = merged_df.groupby('true_class').apply(
        lambda x: (x['predicted_class'] == x['true_class']).mean()
    ).reset_index(name='accuracy')

    print("\n各类别准确率:")
    print(class_accuracy)


if __name__ == "__main__":
    main()