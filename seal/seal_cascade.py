import json
import fitz  # PyMuPDF
import io
import traceback
import argparse
import time
import yaml
from pathlib import Path
from PIL import Image
from paddlex import create_pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
from paddlex.inference.pipelines.base import BasePipeline
from paddlex.inference.pipelines.components import CropByBoxes
from paddlex.inference.pipelines.seal_recognition.result import SealRecognitionResult
from paddlex.inference.common.batch_sampler import ImageBatchSampler
from paddlex.inference.common.reader import ReadImage
from paddlex.inference.models.object_detection.result import DetResult
from paddlex.inference.utils.benchmark import benchmark
from paddlex.inference.utils.hpi import HPIConfig
from paddlex.inference.utils.pp_option import PaddlePredictorOption
from paddlex.utils import logging


def calculate_text_orientation_angle(rec_texts: List[str], textline_orientation_angles: List[int]) -> Tuple[float, Dict]:
    """
    根据文字检测结果和文字方向计算印章的实际旋转角度
    
    Args:
        rec_texts: 识别的文本内容列表
        textline_orientation_angles: 文字行方向角度列表 (0=正常, 1=180度倒置)
    
    Returns:
        - correction_angle: 需要旋转的角度（度，顺时针为正）
    """
    if not rec_texts or not textline_orientation_angles:
        return 0.0, {"normal": 0, "inverted": 0, "total": 0, "confidence": 0.0}
    
    # 只统计有效文本行（过滤空字符串）
    valid_indices = [i for i, text in enumerate(rec_texts) 
                    if i < len(textline_orientation_angles) and text and text.strip()]
    
    if not valid_indices:
        return 0.0, {"normal": 0, "inverted": 0, "total": 0, "confidence": 0.0}
    
    # 统计正常和倒置的文本行数量
    normal_count = sum(1 for i in valid_indices if textline_orientation_angles[i] == 0)
    inverted_count = sum(1 for i in valid_indices if textline_orientation_angles[i] == 1)
    
    # 判断是否需要旋转180度
    # 如果倒置文本占多数，需要旋转180度纠正
    correction_angle = 180.0 if inverted_count > normal_count else 0.0
    
    return correction_angle


def calculate_seal_angle_with_text(
    coordinate: List[float],
    rec_texts: List[str] = None,
    textline_orientation_angles: List[int] = None
) -> Dict[str, any]:
    """
    综合边界框和文字方向计算印章的旋转角度
    
    Args:
        coordinate: 印章边界框坐标 [x0, y0, x1, y1]
        rec_texts: 识别的文本内容列表（可选）
        textline_orientation_angles: 文字行方向角度列表（可选，0=正常，1=180度倒置）
    
    Returns:
        包含角度信息的字典:
        - correction_angle: 纠正角度（顺时针旋转，0或180度）
        - is_rotated: 是否需要旋转（是否倒置）
        - method: 计算方法 ("text" 或 "no_text")
    """
    try:
        result = {
            "correction_angle": 0.0,
            "is_rotated": False,
            "method": "none",
            "center": [0.0, 0.0],
            "size": [0.0, 0.0]
        }
        
        # 计算边界框中心和尺寸
        x0, y0, x1, y1 = coordinate
        center = [(x0 + x1) / 2, (y0 + y1) / 2]
        size = [x1 - x0, y1 - y0]
        result["center"] = center
        result["size"] = size
        
        # 优先使用文字方向判断
        if rec_texts and textline_orientation_angles:
            correction_angle = calculate_text_orientation_angle(rec_texts, textline_orientation_angles)
            
            result.update({
                "correction_angle": correction_angle,
                "is_rotated": correction_angle != 0,
                "method": "text_orientation",
            })
        else:
            # 如果没有文字信息，假设不需要旋转
            result.update({
                "correction_angle": 0.0,
                "is_rotated": False,
                "method": "no_text_data"
            })
        
        return result
        
    except Exception as e:
        logging.warning(f"计算印章角度失败: {e}")
        return {
            "correction_angle": 0.0,
            "is_rotated": False,
            "method": "error",
            "error": str(e),
            "center": [0.0, 0.0],
            "size": [0.0, 0.0]
        }


def rotate_image(image: np.ndarray, angle: float, center: tuple = None) -> np.ndarray:
    """
    旋转图像
    
    Args:
        image: 输入图像
        angle: 旋转角度（顺时针为正，度）
        center: 旋转中心，如果为None则使用图像中心
    
    Returns:
        旋转后的图像
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    
    # 创建旋转矩阵（OpenCV中负角度表示顺时针）
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # 计算旋转后的图像尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # 调整旋转矩阵以适应新尺寸
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # 应用旋转
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))
    
    return rotated


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PDF/图片印章识别和裁剪工具（多模型级联版本）')
    parser.add_argument('--input', type=str, default='招股书.pdf', 
                        help='输入PDF/图片文件或文件夹路径 (默认: 招股书.pdf)')
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
    parser.add_argument('--config', type=str, default='CustomSealRecognition.yaml',
                        help='自定义配置文件路径 (默认: CustomSealRecognition.yaml)')
    return parser.parse_args()


def get_input_files(input_path):
    """获取所有需要处理的文件（PDF和图片）
    
    Args:
        input_path: 输入路径，可以是文件或文件夹
        
    Returns:
        文件路径列表
    """
    # 支持的文件格式
    SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"错误：路径不存在: {input_path}")
        return []
    
    if input_path.is_file():
        # 检查文件扩展名
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        else:
            print(f"错误：不支持的文件格式: {input_path}")
            print(f"支持的格式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            return []
    elif input_path.is_dir():
        # 扫描文件夹中的所有支持的文件
        all_files = []
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(input_path.glob(f'**/*{ext}'))
            # 同时支持大写扩展名
            all_files.extend(input_path.glob(f'**/*{ext.upper()}'))
        
        if not all_files:
            print(f"警告：文件夹中未找到支持的文件: {input_path}")
            print(f"支持的格式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return sorted(set(all_files))  # 使用set去重，然后排序
    else:
        print(f"错误：无效的路径: {input_path}")
        return []


@benchmark.time_methods
class _CustomSealRecognitionPipeline(BasePipeline):
    """自定义印章识别Pipeline - 支持多模型级联检测"""

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """初始化自定义印章识别pipeline"""
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_config=hpi_config
        )

        # 初始化文档预处理pipeline
        self.use_doc_preprocessor = config.get("use_doc_preprocessor", True)
        if self.use_doc_preprocessor:
            doc_preprocessor_config = config.get("SubPipelines", {}).get(
                "DocPreprocessor",
                {
                    "pipeline_config_error": "config error for doc_preprocessor_pipeline!"
                },
            )
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config
            )

        # 初始化多个layout detection模型
        self.use_layout_detection = config.get("use_layout_detection", True)
        self.layout_det_models = []
        
        if self.use_layout_detection:
            # 获取layout detection模型配置列表
            layout_det_configs = config.get("SubModules", {}).get(
                "LayoutDetectionModels", []
            )
            
            if not layout_det_configs:
                raise ValueError("LayoutDetectionModels配置不能为空！")
            
            # 逐个初始化layout detection模型
            for i, layout_config in enumerate(layout_det_configs):
                logging.info(f"初始化第{i+1}个LayoutDetection模型: {layout_config.get('model_name')}")
                
                layout_kwargs = {}
                if (threshold := layout_config.get("threshold", None)) is not None:
                    layout_kwargs["threshold"] = threshold
                if (layout_nms := layout_config.get("layout_nms", None)) is not None:
                    layout_kwargs["layout_nms"] = layout_nms
                if (
                    layout_unclip_ratio := layout_config.get(
                        "layout_unclip_ratio", None
                    )
                ) is not None:
                    layout_kwargs["layout_unclip_ratio"] = layout_unclip_ratio
                if (
                    layout_merge_bboxes_mode := layout_config.get(
                        "layout_merge_bboxes_mode", None
                    )
                ) is not None:
                    layout_kwargs["layout_merge_bboxes_mode"] = layout_merge_bboxes_mode
                
                model = self.create_model(layout_config, **layout_kwargs)
                self.layout_det_models.append({
                    "model": model,
                    "name": layout_config.get("model_name", f"model_{i}"),
                    "config": layout_config
                })
        
        # 初始化印章OCR pipeline
        seal_ocr_config = config.get("SubPipelines", {}).get(
            "SealOCR", {"pipeline_config_error": "config error for seal_ocr_pipeline!"}
        )
        self.seal_ocr_pipeline = self.create_pipeline(seal_ocr_config)

        self._crop_by_boxes = CropByBoxes()
        self.batch_sampler = ImageBatchSampler(batch_size=config.get("batch_size", 1))
        self.img_reader = ReadImage(format="BGR")

    def _has_seal_boxes(self, layout_det_res: DetResult) -> bool:
        """检查layout detection结果中是否包含印章"""
        if not layout_det_res or "boxes" not in layout_det_res:
            return False
        
        for box_info in layout_det_res["boxes"]:
            if box_info["label"].lower() in ["seal"]:
                return True
        
        return False

    def _try_layout_detection(
        self,
        doc_preprocessor_images: List[np.ndarray],
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
    ) -> Tuple[List[DetResult], str]:
        """依次尝试多个layout detection模型，直到找到印章"""
        for model_info in self.layout_det_models:
            model = model_info["model"]
            model_name = model_info["name"]
            
            logging.info(f"尝试使用模型 {model_name} 进行版面检测...")
            
            # 使用当前模型进行检测
            layout_det_results = list(
                model(
                    doc_preprocessor_images,
                    threshold=layout_threshold,
                    layout_nms=layout_nms,
                    layout_unclip_ratio=layout_unclip_ratio,
                    layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                )
            )
            
            # 检查是否检测到印章
            has_seal = any(self._has_seal_boxes(res) for res in layout_det_results)
            
            if has_seal:
                logging.info(f"✓ 模型 {model_name} 成功检测到印章！")
                return layout_det_results, model_name
            else:
                logging.info(f"✗ 模型 {model_name} 未检测到印章，尝试下一个模型...")
        
        # 如果所有模型都没有检测到印章，返回最后一个模型的结果
        logging.warning("⚠️ 所有模型都未能检测到印章，使用最后一个模型的结果")
        return layout_det_results, self.layout_det_models[-1]["name"]

    def check_model_settings_valid(
        self, model_settings: Dict, layout_det_res: DetResult
    ) -> bool:
        """检查模型设置是否有效"""
        if model_settings["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "设置了use_doc_preprocessor，但文档预处理模型未初始化"
            )
            return False

        if model_settings["use_layout_detection"]:
            if layout_det_res is not None:
                logging.error(
                    "layout detection模型已初始化，请设置use_layout_detection=False"
                )
                return False

            if not self.use_layout_detection:
                logging.error(
                    "设置了use_layout_detection，但layout detection模型未初始化"
                )
                return False
        return True

    def get_model_settings(
        self,
        use_doc_orientation_classify: Optional[bool],
        use_doc_unwarping: Optional[bool],
        use_layout_detection: Optional[bool],
    ) -> dict:
        """获取模型设置"""
        if use_doc_orientation_classify is None and use_doc_unwarping is None:
            use_doc_preprocessor = self.use_doc_preprocessor
        else:
            if use_doc_orientation_classify is True or use_doc_unwarping is True:
                use_doc_preprocessor = True
            else:
                use_doc_preprocessor = False

        if use_layout_detection is None:
            use_layout_detection = self.use_layout_detection
        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
            use_layout_detection=use_layout_detection,
        )

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_layout_detection: Optional[bool] = None,
        layout_det_res: Optional[Union[DetResult, List[DetResult]]] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_rec_score_thresh: Optional[float] = None,
        **kwargs,
    ) -> SealRecognitionResult:
        """预测方法"""
        model_settings = self.get_model_settings(
            use_doc_orientation_classify, use_doc_unwarping, use_layout_detection
        )

        if not self.check_model_settings_valid(model_settings, layout_det_res):
            yield {"error": "输入的模型设置参数无效！"}

        external_layout_det_results = layout_det_res
        if external_layout_det_results is not None:
            if not isinstance(external_layout_det_results, list):
                external_layout_det_results = [external_layout_det_results]
            external_layout_det_results = iter(external_layout_det_results)

        for _, batch_data in enumerate(self.batch_sampler(input)):
            image_arrays = self.img_reader(batch_data.instances)

            # 文档预处理
            if model_settings["use_doc_preprocessor"]:
                doc_preprocessor_results = list(
                    self.doc_preprocessor_pipeline(
                        image_arrays,
                        use_doc_orientation_classify=use_doc_orientation_classify,
                        use_doc_unwarping=use_doc_unwarping,
                    )
                )
            else:
                doc_preprocessor_results = [{"output_img": arr} for arr in image_arrays]

            doc_preprocessor_images = [
                item["output_img"] for item in doc_preprocessor_results
            ]

            used_model_name = None
            
            # 版面检测：不使用版面检测或使用外部结果
            if (
                not model_settings["use_layout_detection"]
                and external_layout_det_results is None
            ):
                layout_det_results = [{} for _ in doc_preprocessor_images]
                flat_seal_results = list(
                    self.seal_ocr_pipeline(
                        doc_preprocessor_images,
                        text_det_limit_side_len=seal_det_limit_side_len,
                        text_det_limit_type=seal_det_limit_type,
                        text_det_thresh=seal_det_thresh,
                        text_det_box_thresh=seal_det_box_thresh,
                        text_det_unclip_ratio=seal_det_unclip_ratio,
                        text_rec_score_thresh=seal_rec_score_thresh,
                    )
                )
                for seal_res in flat_seal_results:
                    seal_res["seal_region_id"] = 1
                seal_results = [[item] for item in flat_seal_results]
            else:
                # 使用多模型级联检测
                if model_settings["use_layout_detection"]:
                    layout_det_results, used_model_name = self._try_layout_detection(
                        doc_preprocessor_images,
                        layout_threshold=layout_threshold,
                        layout_nms=layout_nms,
                        layout_unclip_ratio=layout_unclip_ratio,
                        layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                    )
                else:
                    # 使用外部提供的版面检测结果
                    layout_det_results = []
                    for _ in doc_preprocessor_images:
                        try:
                            layout_det_res = next(external_layout_det_results)
                        except StopIteration:
                            raise ValueError("外部版面检测结果数量不足")
                        layout_det_results.append(layout_det_res)

                # 根据版面检测结果裁剪印章区域，并计算印章角度
                cropped_imgs = []
                chunk_indices = [0]
                for doc_preprocessor_image, layout_det_res in zip(
                    doc_preprocessor_images, layout_det_results
                ):
                    for box_info in layout_det_res["boxes"]:
                        if box_info["label"].lower() in ["seal"]:
                            # 裁剪印章区域（角度信息将在OCR后计算）
                            crop_img_info = self._crop_by_boxes(
                                doc_preprocessor_image, [box_info]
                            )
                            crop_img_info = crop_img_info[0]
                            cropped_imgs.append(crop_img_info["img"])
                    chunk_indices.append(len(cropped_imgs))

                # 对裁剪出的印章区域进行OCR识别
                flat_seal_results = list(
                    self.seal_ocr_pipeline(
                        cropped_imgs,
                        text_det_limit_side_len=seal_det_limit_side_len,
                        text_det_limit_type=seal_det_limit_type,
                        text_det_thresh=seal_det_thresh,
                        text_det_box_thresh=seal_det_box_thresh,
                        text_det_unclip_ratio=seal_det_unclip_ratio,
                        text_rec_score_thresh=seal_rec_score_thresh,
                    )
                )

                # 将OCR结果按图像分组
                seal_results = [
                    flat_seal_results[i:j]
                    for i, j in zip(chunk_indices[:-1], chunk_indices[1:])
                ]

                # 为每个印章区域添加ID并计算角度
                for seal_results_for_img, layout_det_res in zip(seal_results, layout_det_results):
                    seal_region_id = 1
                    seal_index = 0
                    
                    # 找到layout_det_res中的印章框
                    seal_boxes = [box for box in layout_det_res.get("boxes", []) if box.get("label", "").lower() == "seal"]
                    
                    for seal_res in seal_results_for_img:
                        seal_res["seal_region_id"] = seal_region_id
                        
                        # 为对应的印章框添加角度信息
                        if seal_index < len(seal_boxes):
                            box_info = seal_boxes[seal_index]
                            
                            # 基于OCR文本方向计算角度
                            angle_info = calculate_seal_angle_with_text(
                                coordinate=box_info["coordinate"],
                                rec_texts=seal_res.get("rec_texts", []),
                                textline_orientation_angles=seal_res.get("textline_orientation_angles", [])
                            )
                            
                            box_info["angle_info"] = angle_info
                        
                        seal_region_id += 1
                        seal_index += 1

            # 构造结果
            for (
                input_path,
                page_index,
                doc_preprocessor_res,
                layout_det_res,
                seal_results_for_img,
            ) in zip(
                batch_data.input_paths,
                batch_data.page_indexes,
                doc_preprocessor_results,
                layout_det_results,
                seal_results,
            ):
                single_img_res = {
                    "input_path": input_path,
                    "page_index": page_index,
                    "doc_preprocessor_res": doc_preprocessor_res,
                    "layout_det_res": layout_det_res,
                    "seal_res_list": seal_results_for_img,
                    "model_settings": model_settings,
                }
                # 添加使用的模型信息
                if used_model_name:
                    single_img_res["used_layout_model"] = used_model_name
                
                yield SealRecognitionResult(single_img_res)


# 使用线程本地存储为每个线程创建独立的 pipeline
thread_local = threading.local()


def get_pipeline(args):
    """获取当前线程的 pipeline 实例（线程安全）"""
    if not hasattr(thread_local, 'pipeline'):
        print_lock = threading.Lock()
        with print_lock:
            print(f"[线程 {threading.current_thread().name}] 初始化自定义多模型级联 pipeline...")
        
        # 读取配置文件
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建自定义pipeline实例
        thread_local.pipeline = _CustomSealRecognitionPipeline(config=config)
        
    return thread_local.pipeline


def is_image_file(file_path):
    """判断文件是否为图片格式"""
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def process_file(file_path, output_dir, args):
    """处理单个文件（PDF或图片）"""
    print("\n" + "=" * 80)
    print(f"开始处理: {file_path}")
    print(f"文件类型: {'图片' if is_image_file(file_path) else 'PDF'}")
    print(f"使用配置: {args.config}")
    print("=" * 80)
    
    file_start_time = time.time()
    
    # 创建临时图片目录
    temp_images_dir = output_dir / "temp_images"
    temp_images_dir.mkdir(exist_ok=True)
    
    # ==================== 第一步：准备图片（PDF转图片或直接使用图片）====================
    step1_start_time = time.time()
    print("\n" + "=" * 60)
    
    if is_image_file(file_path):
        # 图片文件，直接使用
        print("第一步：准备图片文件")
        print("=" * 60)
        print(f"检测到图片文件，直接使用...")
        
        # 读取图片信息
        img = Image.open(file_path)
        img_width, img_height = img.size
        
        print(f"\n图片信息:")
        print(f"  - 图片尺寸: {img_width} x {img_height} pixels")
        print(f"  - 图片格式: {img.format}")
        
        # 将图片复制到临时目录（统一命名为page_0.png）
        image_path = temp_images_dir / f"page_0{file_path.suffix}"
        img.save(str(image_path))
        
        page_images = [image_path]
        total_pages = 1
        
        step1_end_time = time.time()
        step1_duration = step1_end_time - step1_start_time
        print(f"\n✓ 图片准备完成")
        print(f"⏱️  第一步耗时: {step1_duration:.2f} 秒\n")
    else:
        # PDF文件，需要转换为图片
        print("第一步：将PDF每一页导出为图片（多线程并行）")
        print("=" * 60)
        
        pdf_document = fitz.open(str(file_path))
        total_pages = len(pdf_document)
        pdf_document.close()
        
        # 创建线程锁用于保护共享资源
        export_lock = threading.Lock()
        
        def export_page(page_num):
            """导出单个页面为图片"""
            try:
                # 每个线程需要独立打开PDF
                doc = fitz.open(str(file_path))
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
    
    # ==================== 第二步：使用多模型级联识别图片中的印章 ====================
    step2_start_time = time.time()
    print("=" * 60)
    print("第二步：使用多模型级联识别图片中的印章（多线程并行）")
    print("=" * 60)
    
    # 创建线程锁用于保护共享资源
    print_lock = threading.Lock()
    results_lock = threading.Lock()
    
    # 定义处理单个图片的函数
    def process_image(page_num, image_path):
        """处理单个图片的印章识别"""
        try:
            with print_lock:
                print(f"\n[线程 {threading.current_thread().name}] 开始处理: {image_path.name}")
            
            # 为每个线程获取独立的 pipeline 实例
            pipeline = get_pipeline(args)
            
            output = pipeline.predict(
                str(image_path),
                use_doc_orientation_classify=args.use_orientation,
                use_doc_unwarping=args.use_unwarping,
            )
            
            page_results = []
            for res in output:
                with print_lock:
                    res.print()  ## 打印预测的结构化输出
                    # 打印使用的模型信息
                    if hasattr(res, '_dict') and "used_layout_model" in res._dict:
                        print(f"  🎯 使用的版面检测模型: {res._dict['used_layout_model']}")
                
                res.save_to_img(str(output_dir) + "/")  ## 保存可视化结果
                res.save_to_json(str(output_dir) + "/")  ## 保存预测结果为JSON
                page_results.append({
                    'page_num': page_num,
                    'image_path': image_path,
                    'result': res,
                    'used_model': res._dict.get('used_layout_model', 'unknown') if hasattr(res, '_dict') else 'unknown'
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
    print(f"\n✓ 识别完成！结果已保存到 {output_dir} 目录")
    print(f"⏱️  第二步耗时: {step2_duration:.2f} 秒\n")
    
    # 统计使用的模型
    model_usage = {}
    for res in all_results:
        model_name = res.get('used_model', 'unknown')
        model_usage[model_name] = model_usage.get(model_name, 0) + 1
    
    if model_usage:
        print("\n📊 模型使用统计:")
        for model_name, count in model_usage.items():
            print(f"  - {model_name}: {count} 次")
    
    # ==================== 第三步：从导出的图片中裁剪印章 ====================
    step3_start_time = time.time()
    print("\n" + "=" * 60)
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
                angle_info = box.get("angle_info", {})
                
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
                
                # 显示角度信息
                if angle_info:
                    rot_angle = angle_info.get('rotation_angle', 0)
                    corr_angle = angle_info.get('correction_angle', 0)
                    print(f"    - 旋转角度: {rot_angle:.2f}° (逆时针)")
                    print(f"    - 纠正角度: {corr_angle:.2f}° (顺时针)")
                    if angle_info.get('is_rotated', False):
                        print(f"    - 状态: 有旋转")
                    else:
                        print(f"    - 状态: 无旋转（正向）")
                
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
                    
                    # 检查是否需要旋转纠正
                    if angle_info and angle_info.get('is_rotated', False):
                        correction_angle = angle_info.get('correction_angle', 0)
                        print(f"    - 应用旋转纠正: {correction_angle}°")
                        
                        # 将PIL图片转换为numpy数组
                        seal_img_np = np.array(seal_img)
                        
                        # 应用旋转
                        seal_img_rotated = rotate_image(seal_img_np, correction_angle)
                        
                        # 转换回PIL图片
                        seal_img = Image.fromarray(seal_img_rotated)
                        print(f"    - 纠正后尺寸: {seal_img.width}x{seal_img.height} pixels")
                    
                    # 保存原始裁剪图片（未纠正）
                    output_filename_orig = f"seal_cropped_page{page_num}_{seal_count}_score{score:.3f}_orig.png"
                    output_path_orig = output_dir / output_filename_orig
                    page_image.crop((x0, y0, x1, y1)).save(str(output_path_orig))
                    
                    # 保存纠正后的图片
                    output_filename = f"seal_cropped_page{page_num}_{seal_count}_score{score:.3f}.png"
                    output_path = output_dir / output_filename
                    seal_img.save(str(output_path))
                    
                    print(f"    ✓ 已保存原始: {output_path_orig}")
                    print(f"    ✓ 已保存纠正后: {output_path}")
                    print(f"    - 最终尺寸: {seal_img.width}x{seal_img.height} pixels")
                    
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
    file_end_time = time.time()
    file_duration = file_end_time - file_start_time
    
    print("\n" + "=" * 60)
    print("⏱️  耗时统计")
    print("=" * 60)
    step1_name = "图片准备" if is_image_file(file_path) else "PDF导出图片"
    print(f"  第一步（{step1_name}）: {step1_duration:.2f} 秒 ({step1_duration/file_duration*100:.1f}%)")
    print(f"  第二步（印章识别）  : {step2_duration:.2f} 秒 ({step2_duration/file_duration*100:.1f}%)")
    print(f"  第三步（印章裁剪）  : {step3_duration:.2f} 秒 ({step3_duration/file_duration*100:.1f}%)")
    print("-" * 60)
    print(f"  总耗时              : {file_duration:.2f} 秒")
    print("=" * 60)
    
    return {
        'file_path': file_path,
        'file_type': '图片' if is_image_file(file_path) else 'PDF',
        'total_pages': total_pages,
        'total_seals': total_seals,
        'step1_duration': step1_duration,
        'step2_duration': step2_duration,
        'step3_duration': step3_duration,
        'total_duration': file_duration,
        'model_usage': model_usage
    }


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 记录程序开始时间
    program_start_time = time.time()
    
    print("\n" + "=" * 80)
    print("🔖 多模型级联印章识别工具")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"并发线程数: {args.max_workers}")
    print(f"文档方向分类: {'开启' if args.use_orientation else '关闭'}")
    print(f"文档矫正: {'开启' if args.use_unwarping else '关闭'}")
    print("=" * 80)
    
    # 获取所有需要处理的文件
    input_files = get_input_files(args.input)
    
    if not input_files:
        print("错误：没有找到需要处理的文件")
        exit(1)
    
    # 统计文件类型
    pdf_count = sum(1 for f in input_files if not is_image_file(f))
    image_count = sum(1 for f in input_files if is_image_file(f))
    
    print("\n" + "=" * 80)
    print(f"找到 {len(input_files)} 个文件待处理 (PDF: {pdf_count}, 图片: {image_count})")
    print("=" * 80)
    for i, input_file in enumerate(input_files, 1):
        file_type = "图片" if is_image_file(input_file) else "PDF"
        print(f"  {i}. [{file_type}] {input_file}")
    print("=" * 80)
    
    # 创建主输出目录
    base_output_dir = Path(args.output)
    base_output_dir.mkdir(exist_ok=True)
    
    # 处理每个文件
    all_stats = []
    for input_file in input_files:
        # 为每个文件创建独立的输出目录
        # 使用文件名（不含扩展名）作为子目录名
        file_output_dir = base_output_dir / input_file.stem
        file_output_dir.mkdir(exist_ok=True)
        
        try:
            # 处理单个文件
            stats = process_file(input_file, file_output_dir, args)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n✗ 处理文件失败: {input_file}")
            print(f"  错误信息: {e}")
            traceback.print_exc()
    
    # 计算并显示总体统计
    program_end_time = time.time()
    total_program_duration = program_end_time - program_start_time
    
    print("\n\n" + "=" * 80)
    print("📊 总体统计")
    print("=" * 80)
    print(f"  处理文件数      : {len(all_stats)}/{len(input_files)}")
    
    if all_stats:
        total_pages = sum(s['total_pages'] for s in all_stats)
        total_seals = sum(s['total_seals'] for s in all_stats)
        avg_duration = sum(s['total_duration'] for s in all_stats) / len(all_stats)
        
        # 统计已处理的文件类型
        processed_pdf_count = sum(1 for s in all_stats if s['file_type'] == 'PDF')
        processed_image_count = sum(1 for s in all_stats if s['file_type'] == '图片')
        
        print(f"  - PDF文件       : {processed_pdf_count}")
        print(f"  - 图片文件      : {processed_image_count}")
        print(f"  总页数          : {total_pages}")
        print(f"  总印章数        : {total_seals}")
        print(f"  平均处理时间    : {avg_duration:.2f} 秒/文件")
        
        # 统计所有文件的模型使用情况
        total_model_usage = {}
        for stats in all_stats:
            for model_name, count in stats.get('model_usage', {}).items():
                total_model_usage[model_name] = total_model_usage.get(model_name, 0) + count
        
        if total_model_usage:
            print("\n  🎯 模型使用统计:")
            for model_name, count in sorted(total_model_usage.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {model_name}: {count} 次")
    
    print(f"\n  程序总耗时      : {total_program_duration:.2f} 秒")
    print("=" * 80)
    print("✓ 全部处理完成！")
    print("=" * 80)

