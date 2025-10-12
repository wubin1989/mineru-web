import json
import fitz  # PyMuPDF
import io
import traceback
import argparse
import time
from pathlib import Path
from PIL import Image
from paddlex import create_pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PDF印章识别和裁剪工具')
    parser.add_argument('--pdf', type=str, default='招股书.pdf', 
                        help='输入PDF文件路径 (默认: 招股书.pdf)')
    parser.add_argument('--output', type=str, default='./output', 
                        help='输出目录路径 (默认: ./output)')
    parser.add_argument('--zoom', type=float, default=2.0, 
                        help='PDF转图片的缩放倍数 (默认: 2.0)')
    parser.add_argument('--max-workers', type=int, default=4, 
                        help='并发处理的最大线程数 (默认: 4)')
    parser.add_argument('--use-orientation', action='store_true', 
                        help='是否使用文档方向分类 (默认: False)')
    parser.add_argument('--use-unwarping', action='store_true', 
                        help='是否使用文档矫正 (默认: False)')
    return parser.parse_args()

# 解析命令行参数
args = parse_args()

# 记录程序开始时间
program_start_time = time.time()

pdf_path = args.pdf
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True)

# 创建临时图片目录
temp_images_dir = output_dir / "temp_pdf_images"
temp_images_dir.mkdir(exist_ok=True)

# ==================== 第一步：将PDF每一页导出为图片（多线程并行）====================
step1_start_time = time.time()
print("=" * 60)
print("第一步：将PDF每一页导出为图片（多线程并行）")
print("=" * 60)

pdf_document = fitz.open(pdf_path)
total_pages = len(pdf_document)
pdf_document.close()

# 创建线程锁用于保护共享资源
export_lock = threading.Lock()

def export_page(page_num):
    """导出单个页面为图片"""
    try:
        # 每个线程需要独立打开PDF
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # 使用指定的分辨率导出图片
        mat = fitz.Matrix(args.zoom, args.zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # 保存为PNG图片
        image_filename = f"page_{page_num}.png"
        image_path = temp_images_dir / image_filename
        pix.save(str(image_path))
        
        # 获取页面和图片信息
        pdf_width = page.rect.width
        pdf_height = page.rect.height
        img_width = pix.width
        img_height = pix.height
        
        doc.close()
        
        with export_lock:
            print(f"页面 {page_num}: 已导出 {image_path}")
            print(f"  - PDF尺寸: {pdf_width:.2f} x {pdf_height:.2f}")
            print(f"  - 图片尺寸: {img_width} x {img_height} pixels")
            print()
        
        return image_path
    except Exception as e:
        with export_lock:
            print(f"✗ 导出页面 {page_num} 失败: {e}")
            traceback.print_exc()
        return None

# 使用线程池并发导出所有页面
page_images = [None] * total_pages  # 预分配列表保持页面顺序
max_export_workers = min(args.max_workers, total_pages)

print(f"使用 {max_export_workers} 个线程并发导出 {total_pages} 页图片...\n")

with ThreadPoolExecutor(max_workers=max_export_workers) as executor:
    # 提交所有任务
    future_to_page = {
        executor.submit(export_page, page_num): page_num
        for page_num in range(total_pages)
    }
    
    # 收集结果
    for future in as_completed(future_to_page):
        page_num = future_to_page[future]
        try:
            image_path = future.result()
            if image_path:
                page_images[page_num] = image_path
        except Exception as e:
            with export_lock:
                print(f"✗ 获取页面 {page_num} 导出结果时出错: {e}")

# 过滤掉失败的页面
page_images = [img for img in page_images if img is not None]

step1_end_time = time.time()
step1_duration = step1_end_time - step1_start_time
print(f"\n✓ 共成功导出 {len(page_images)} 页图片")
print(f"⏱️  第一步耗时: {step1_duration:.2f} 秒\n")

# ==================== 第二步：使用 PaddleX 识别图片中的印章 ====================
step2_start_time = time.time()
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
        thread_local.pipeline = create_pipeline(pipeline="SealRecognition.yaml")
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
            use_doc_orientation_classify=args.use_orientation,
            use_doc_unwarping=args.use_unwarping,
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
max_workers = min(args.max_workers, len(page_images))  # 根据参数设置线程数，避免过度并发

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

step2_end_time = time.time()
step2_duration = step2_end_time - step2_start_time
print(f"\n✓ 识别完成！结果已保存到 ./output/ 目录")
print(f"⏱️  第二步耗时: {step2_duration:.2f} 秒\n")

# ==================== 第三步：从导出的图片中裁剪印章 ====================
step3_start_time = time.time()
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

step3_end_time = time.time()
step3_duration = step3_end_time - step3_start_time

print("\n" + "=" * 60)
if total_seals == 0:
    print("未找到任何印章")
else:
    print(f"全部完成！总共裁剪了 {total_seals} 个印章")
print(f"⏱️  第三步耗时: {step3_duration:.2f} 秒")
print("=" * 60)

# 计算并显示总耗时
program_end_time = time.time()
total_duration = program_end_time - program_start_time

print("\n" + "=" * 60)
print("⏱️  耗时统计")
print("=" * 60)
print(f"  第一步（PDF导出图片）: {step1_duration:.2f} 秒 ({step1_duration/total_duration*100:.1f}%)")
print(f"  第二步（印章识别）  : {step2_duration:.2f} 秒 ({step2_duration/total_duration*100:.1f}%)")
print(f"  第三步（印章裁剪）  : {step3_duration:.2f} 秒 ({step3_duration/total_duration*100:.1f}%)")
print("-" * 60)
print(f"  总耗时              : {total_duration:.2f} 秒")
print("=" * 60)