import json
import fitz  # PyMuPDF
import io
import traceback
from pathlib import Path
from PIL import Image
from paddlex import create_pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

pdf_path = "招股书.pdf"
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

# 创建临时图片目录
temp_images_dir = output_dir / "temp_pdf_images"
temp_images_dir.mkdir(exist_ok=True)

# ==================== 第一步：将PDF每一页导出为图片 ====================
print("=" * 60)
print("第一步：将PDF每一页导出为图片")
print("=" * 60)

pdf_document = fitz.open(pdf_path)
page_images = []  # 保存导出的图片路径

for page_num in range(len(pdf_document)):
    page = pdf_document[page_num]
    
    # 使用较高的分辨率导出图片（2倍缩放）
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    # 保存为PNG图片
    image_filename = f"page_{page_num}.png"
    image_path = temp_images_dir / image_filename
    pix.save(str(image_path))
    page_images.append(image_path)
    
    print(f"页面 {page_num}: 已导出 {image_path}")
    print(f"  - PDF尺寸: {page.rect.width:.2f} x {page.rect.height:.2f}")
    print(f"  - 图片尺寸: {pix.width} x {pix.height} pixels")
    print()

pdf_document.close()

print(f"✓ 共导出 {len(page_images)} 页图片\n")

# ==================== 第二步：使用 PaddleX 识别图片中的印章 ====================
print("=" * 60)
print("第二步：使用 PaddleX 识别图片中的印章（多线程并发）")
print("=" * 60)

# 创建线程锁用于保护共享资源
print_lock = threading.Lock()
results_lock = threading.Lock()

# 使用线程本地存储为每个线程创建独立的 pipeline
thread_local = threading.local()

def get_pipeline():
    """获取当前线程的 pipeline 实例（线程安全）"""
    if not hasattr(thread_local, 'pipeline'):
        with print_lock:
            print(f"[线程 {threading.current_thread().name}] 初始化 pipeline...")
        thread_local.pipeline = create_pipeline(pipeline="seal_recognition")
    return thread_local.pipeline

# 定义处理单个图片的函数
def process_image(page_num, image_path):
    """处理单个图片的印章识别"""
    try:
        with print_lock:
            print(f"\n[线程 {threading.current_thread().name}] 开始处理: {image_path.name}")
        
        # 为每个线程获取独立的 pipeline 实例
        pipeline = get_pipeline()
        
        output = pipeline.predict(
            str(image_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        
        page_results = []
        for res in output:
            with print_lock:
                res.print()  ## 打印预测的结构化输出
            res.save_to_img(str(output_dir) + "/")  ## 保存可视化结果
            res.save_to_json(str(output_dir) + "/")  ## 保存预测结果为JSON
            page_results.append({
                'page_num': page_num,
                'image_path': image_path,
                'result': res
            })
        
        with print_lock:
            print(f"✓ [线程 {threading.current_thread().name}] 完成处理: {image_path.name}")
        
        return page_results
    
    except Exception as e:
        with print_lock:
            print(f"✗ [线程 {threading.current_thread().name}] 处理失败: {image_path.name}")
            print(f"  错误信息: {e}")
            traceback.print_exc()
        return []

# 使用线程池并发处理所有图片
all_results = []
max_workers = min(4, len(page_images))  # 最多使用4个线程，避免过度并发

print(f"\n使用 {max_workers} 个线程并发处理 {len(page_images)} 页图片...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交所有任务
    future_to_page = {
        executor.submit(process_image, page_num, image_path): (page_num, image_path)
        for page_num, image_path in enumerate(page_images)
    }
    
    # 收集结果
    for future in as_completed(future_to_page):
        page_num, image_path = future_to_page[future]
        try:
            page_results = future.result()
            with results_lock:
                all_results.extend(page_results)
        except Exception as e:
            with print_lock:
                print(f"✗ 获取页面 {page_num} 结果时出错: {e}")

# 按页码排序结果
all_results.sort(key=lambda x: x['page_num'])

print("\n✓ 识别完成！结果已保存到 ./output/ 目录\n")

# ==================== 第三步：从导出的图片中裁剪印章 ====================
print("=" * 60)
print("第三步：从导出的图片中裁剪印章")
print("=" * 60)

total_seals = 0

# 遍历所有导出的图片
for page_num, image_path in enumerate(page_images):
    # 读取对应的JSON结果文件
    json_filename = f"{image_path.stem}_res.json"
    json_path = output_dir / json_filename
    
    if not json_path.exists():
        print(f"警告：未找到页面 {page_num} 的JSON结果文件: {json_path}")
        continue
    
    # 加载JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 打开图片
    page_image = Image.open(image_path)
    img_width, img_height = page_image.size
    
    print(f"\n页面 {page_num} ({image_path.name}):")
    print(f"  - 图片尺寸: {img_width} x {img_height} pixels")
    
    # 遍历所有检测到的框
    boxes = data.get("layout_det_res", {}).get("boxes", [])
    seal_count = 0
    
    for box in boxes:
        label = box.get("label", "")
        
        # 只处理印章
        if label == "seal":
            seal_count += 1
            coordinate = box["coordinate"]
            score = box["score"]
            
            # coordinate 格式: [x0, y0, x1, y1]
            x0, y0, x1, y1 = coordinate
            
            # 确保坐标有效
            if x0 >= x1 or y0 >= y1:
                print(f"  警告：印章 {seal_count} 坐标无效: [{x0}, {y0}, {x1}, {y1}]，跳过")
                seal_count -= 1
                continue
            
            print(f"  印章 {seal_count}:")
            print(f"    - 坐标: [{x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}]")
            print(f"    - 置信度: {score:.4f}")
            
            try:
                # 转换为整数坐标并确保在图片范围内
                x0 = int(max(0, min(x0, img_width)))
                y0 = int(max(0, min(y0, img_height)))
                x1 = int(max(0, min(x1, img_width)))
                y1 = int(max(0, min(y1, img_height)))
                
                # 计算裁剪区域的宽高
                crop_width = x1 - x0
                crop_height = y1 - y0
                
                if crop_width <= 0 or crop_height <= 0:
                    print(f"    - 错误：裁剪尺寸无效 ({crop_width}x{crop_height})，跳过")
                    seal_count -= 1
                    continue
                
                # 从图片中裁剪印章区域
                seal_img = page_image.crop((x0, y0, x1, y1))
                
                # 保存图片
                output_filename = f"seal_cropped_page{page_num}_{seal_count}_score{score:.3f}.png"
                output_path = output_dir / output_filename
                seal_img.save(str(output_path))
                
                print(f"    ✓ 已保存: {output_path}")
                print(f"    - 尺寸: {seal_img.width}x{seal_img.height} pixels")
                
                total_seals += 1
                
            except Exception as e:
                print(f"    - 错误：保存印章图片时出错: {e}")
                traceback.print_exc()
                seal_count -= 1
    
    if seal_count == 0:
        print(f"  未找到印章")
    else:
        print(f"  ✓ 页面 {page_num} 共裁剪 {seal_count} 个印章")

print("\n" + "=" * 60)
if total_seals == 0:
    print("未找到任何印章")
else:
    print(f"全部完成！总共裁剪了 {total_seals} 个印章")
print("=" * 60)