# 自定义印章识别Pipeline开发文档

> **作者**: wubin  
> **日期**: 2025-10-20

## 一、需求背景

在使用PaddleX的印章识别（seal_recognition）pipeline时，可能会遇到单一的版面检测（LayoutDetection）模型无法在所有场景下准确识别印章的情况。为了提高印章检测的鲁棒性，我们需要实现一个多模型级联尝试的机制：

1. 首先使用 `PP-DocLayout-L` 模型识别印章
2. 如果没有识别到印章，则使用 `PP-DocLayout_plus-L` 模型再次识别
3. 如果还是没有识别到印章，则使用 `RT-DETR-H_layout_3cls` 模型进行识别
4. 一旦识别到印章，立即进行后续的OCR识别步骤

## 二、实现方案概述

### 2.1 技术架构

自定义pipeline的实现主要包括以下几个部分：

```
自定义Pipeline架构
├── 继承BasePipeline基类
├── 初始化多个LayoutDetection模型
├── 实现级联检测逻辑
└── 复用原有的OCR识别功能
```

### 2.2 核心思路

1. **继承现有Pipeline**: 基于 `_SealRecognitionPipeline` 创建自定义类
2. **多模型初始化**: 在 `__init__` 中初始化多个layout detection模型
3. **级联检测逻辑**: 在 `predict` 方法中依次尝试各个模型
4. **结果判断**: 通过检查结果中是否包含"seal"标签来判断是否检测到印章

## 三、实现步骤

### 3.1 创建自定义Pipeline类

在 `/Users/doudou/workspace/PaddleX/paddlex/inference/pipelines/seal_recognition/` 目录下创建新文件 `custom_pipeline.py`:

```python
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ....utils import logging
from ....utils.deps import pipeline_requires_extra
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...models.object_detection.result import DetResult
from ...utils.benchmark import benchmark
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline
from ..components import CropByBoxes
from .result import SealRecognitionResult


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
        """初始化自定义印章识别pipeline
        
        Args:
            config (Dict): 配置字典
            device (str, optional): 运行设备. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor选项. Defaults to None.
            use_hpip (bool, optional): 是否使用高性能推理. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional): 高性能推理配置. Defaults to None.
        """
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
        """检查layout detection结果中是否包含印章
        
        Args:
            layout_det_res (DetResult): 版面检测结果
            
        Returns:
            bool: 如果检测到印章返回True，否则返回False
        """
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
        """依次尝试多个layout detection模型，直到找到印章
        
        Args:
            doc_preprocessor_images: 文档预处理后的图像列表
            layout_threshold: 检测阈值
            layout_nms: 是否使用NMS
            layout_unclip_ratio: unclip比率
            layout_merge_bboxes_mode: 边界框合并模式
            
        Returns:
            Tuple[List[DetResult], str]: (检测结果列表, 使用的模型名称)
        """
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
                logging.info(f"模型 {model_name} 成功检测到印章！")
                return layout_det_results, model_name
            else:
                logging.info(f"模型 {model_name} 未检测到印章，尝试下一个模型...")
        
        # 如果所有模型都没有检测到印章，返回最后一个模型的结果
        logging.warning("所有模型都未能检测到印章，使用最后一个模型的结果")
        return layout_det_results, self.layout_det_models[-1]["name"]

    def check_model_settings_valid(
        self, model_settings: Dict, layout_det_res: DetResult
    ) -> bool:
        """检查模型设置是否有效
        
        Args:
            model_settings (Dict): 模型设置字典
            layout_det_res (DetResult): Layout检测结果
            
        Returns:
            bool: 如果设置有效返回True，否则返回False
        """
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
        """获取模型设置
        
        Args:
            use_doc_orientation_classify: 是否使用文档方向分类
            use_doc_unwarping: 是否使用文档矫正
            use_layout_detection: 是否使用版面检测
            
        Returns:
            dict: 模型设置字典
        """
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
        """预测方法
        
        Args:
            input: 输入图像路径或numpy数组
            use_doc_orientation_classify: 是否使用文档方向分类
            use_doc_unwarping: 是否使用文档矫正
            use_layout_detection: 是否使用版面检测
            layout_det_res: 外部提供的版面检测结果
            layout_threshold: 检测阈值
            layout_nms: 是否使用NMS
            layout_unclip_ratio: unclip比率
            layout_merge_bboxes_mode: 边界框合并模式
            seal_det_limit_side_len: 印章检测限制边长
            seal_det_limit_type: 印章检测限制类型
            seal_det_thresh: 印章检测阈值
            seal_det_box_thresh: 印章检测框阈值
            seal_det_unclip_ratio: 印章检测unclip比率
            seal_rec_score_thresh: 印章识别分数阈值
            
        Yields:
            SealRecognitionResult: 印章识别结果
        """
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
                        threshold=layout_threshold,
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

                # 根据版面检测结果裁剪印章区域
                cropped_imgs = []
                chunk_indices = [0]
                for doc_preprocessor_image, layout_det_res in zip(
                    doc_preprocessor_images, layout_det_results
                ):
                    for box_info in layout_det_res["boxes"]:
                        if box_info["label"].lower() in ["seal"]:
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

                # 为每个印章区域添加ID
                for seal_results_for_img in seal_results:
                    seal_region_id = 1
                    for seal_res in seal_results_for_img:
                        seal_res["seal_region_id"] = seal_region_id
                        seal_region_id += 1

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


@pipeline_requires_extra("ocr")
class CustomSealRecognitionPipeline(AutoParallelImageSimpleInferencePipeline):
    """自定义印章识别Pipeline - 支持多模型级联检测"""
    
    entities = ["custom_seal_recognition"]

    @property
    def _pipeline_cls(self):
        return _CustomSealRecognitionPipeline

    def _get_batch_size(self, config):
        return config.get("batch_size", 1)
```

### 3.2 创建配置文件

在 `/Users/doudou/workspace/PaddleX/paddlex/configs/pipelines/` 目录下创建 `custom_seal_recognition.yaml`:

```yaml
pipeline_name: custom_seal_recognition

use_doc_preprocessor: True
use_layout_detection: True

SubModules:
  # 配置多个LayoutDetection模型，按优先级排列
  LayoutDetectionModels:
    - module_name: layout_detection
      model_name: PP-DocLayout-L
      model_dir: null
      threshold: 0.5
      layout_nms: True
      layout_unclip_ratio: 1.0
      layout_merge_bboxes_mode: "large"
    
    - module_name: layout_detection
      model_name: PP-DocLayout_plus-L
      model_dir: null
      threshold: 0.5
      layout_nms: True
      layout_unclip_ratio: 1.0
      layout_merge_bboxes_mode: "large"
    
    - module_name: layout_detection
      model_name: RT-DETR-H_layout_3cls
      model_dir: null
      threshold: 0.5
      layout_nms: True
      layout_unclip_ratio: 1.0
      layout_merge_bboxes_mode: "large"

SubPipelines:
  DocPreprocessor:
    pipeline_name: doc_preprocessor
    use_doc_orientation_classify: True
    use_doc_unwarping: True
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null
      DocUnwarping:
        module_name: image_unwarping
        model_name: UVDoc
        model_dir: null
  
  SealOCR:
    pipeline_name: OCR
    text_type: seal
    use_doc_preprocessor: False
    use_textline_orientation: False
    SubModules:
      TextDetection:
        module_name: seal_text_detection
        model_name: PP-OCRv4_server_seal_det
        model_dir: null
        limit_side_len: 736
        limit_type: min
        max_side_len: 4000
        thresh: 0.2
        box_thresh: 0.6
        unclip_ratio: 0.5
      TextRecognition:
        module_name: text_recognition
        model_name: PP-OCRv4_server_rec
        model_dir: null
        batch_size: 1
        score_thresh: 0
```

### 3.3 注册自定义Pipeline

修改 `/Users/doudou/workspace/PaddleX/paddlex/inference/pipelines/seal_recognition/__init__.py`:

```python
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .pipeline import SealRecognitionPipeline
from .custom_pipeline import CustomSealRecognitionPipeline  # 新增

__all__ = ["SealRecognitionPipeline", "CustomSealRecognitionPipeline"]  # 新增导出
```

## 四、使用方法

### 4.1 Python脚本方式

创建测试脚本 `test_custom_seal_recognition.py`:

```python
from paddlex import create_pipeline

# 方式1: 使用配置文件
pipeline = create_pipeline(
    pipeline="custom_seal_recognition",
    device="gpu:0"
)

# 方式2: 使用本地配置文件路径
# pipeline = create_pipeline(
#     pipeline="./paddlex/configs/pipelines/custom_seal_recognition.yaml",
#     device="gpu:0"
# )

# 执行预测
output = pipeline.predict("your_document_image.jpg")

# 处理结果
for res in output:
    # 打印结果
    res.print()
    
    # 保存可视化结果
    res.save_to_img("./output/")
    
    # 保存JSON结果
    res.save_to_json("./output/result.json")
    
    # 查看使用了哪个模型
    if "used_layout_model" in res:
        print(f"使用的版面检测模型: {res['used_layout_model']}")
```

### 4.2 命令行方式

```bash
# 使用自定义pipeline
paddlex --pipeline custom_seal_recognition \
        --input your_document_image.jpg \
        --device gpu:0

# 或使用配置文件路径
paddlex --pipeline ./paddlex/configs/pipelines/custom_seal_recognition.yaml \
        --input your_document_image.jpg \
        --device gpu:0
```

### 4.3 批量处理示例

```python
from paddlex import create_pipeline
import os

# 创建pipeline
pipeline = create_pipeline(
    pipeline="custom_seal_recognition",
    device="gpu:0"
)

# 批量处理
image_dir = "./test_images"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

for image_name in os.listdir(image_dir):
    if image_name.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_dir, image_name)
        
        print(f"处理图像: {image_name}")
        output = pipeline.predict(image_path)
        
        for idx, res in enumerate(output):
            # 保存结果
            output_prefix = os.path.join(output_dir, f"{image_name}_result")
            res.save_to_img(output_prefix)
            res.save_to_json(f"{output_prefix}.json")
            
            # 打印使用的模型信息
            if "used_layout_model" in res:
                print(f"  - 使用的模型: {res['used_layout_model']}")
            
            # 打印印章数量
            seal_count = len(res.get("seal_res_list", []))
            print(f"  - 检测到的印章数量: {seal_count}")
```

## 五、配置说明

### 5.1 模型配置参数

在配置文件的 `LayoutDetectionModels` 部分，可以配置多个版面检测模型，每个模型支持以下参数：

| 参数名称 | 类型 | 说明 | 默认值 |
|---------|------|------|--------|
| module_name | str | 模块名称，固定为"layout_detection" | - |
| model_name | str | 模型名称 | - |
| model_dir | str | 本地模型路径（可选） | null |
| threshold | float | 检测阈值 | 0.5 |
| layout_nms | bool | 是否使用NMS | True |
| layout_unclip_ratio | float | unclip比率 | 1.0 |
| layout_merge_bboxes_mode | str | 边界框合并模式 | "large" |

### 5.2 模型尝试顺序

模型按照在 `LayoutDetectionModels` 列表中的顺序依次尝试。建议的配置顺序：

1. **PP-DocLayout-L**: 通用版面检测模型，速度快
2. **PP-DocLayout_plus-L**: 增强版版面检测模型，精度更高
3. **RT-DETR-H_layout_3cls**: 高精度检测模型，计算量较大

可以根据实际需求调整模型顺序和数量。

### 5.3 性能优化建议

1. **模型选择**: 将最常用、成功率最高的模型放在前面
2. **阈值调整**: 适当降低阈值可以提高召回率，但可能增加误检
3. **设备配置**: 使用GPU可以显著提升推理速度
4. **批处理**: 对于大量图像，可以使用批处理提高效率

## 六、高级定制

### 6.1 添加更多模型

如需添加更多版面检测模型，只需在配置文件中添加新的模型配置：

```yaml
SubModules:
  LayoutDetectionModels:
    - module_name: layout_detection
      model_name: 你的模型名称
      model_dir: null
      threshold: 0.5
      # ... 其他参数
```

### 6.2 自定义检测逻辑

如果需要更复杂的检测逻辑（例如基于置信度选择、多模型结果融合等），可以修改 `_try_layout_detection` 方法：

```python
def _try_layout_detection(self, doc_preprocessor_images, **kwargs):
    """自定义检测逻辑"""
    all_results = []
    
    # 收集所有模型的结果
    for model_info in self.layout_det_models:
        results = model_info["model"](doc_preprocessor_images, **kwargs)
        all_results.append((results, model_info["name"]))
    
    # 自定义选择逻辑
    # 例如：选择置信度最高的结果
    # 或者：融合多个模型的结果
    
    return best_results, best_model_name
```

### 6.3 添加日志和监控

可以在关键位置添加日志记录：

```python
# 在__init__中
logging.info(f"初始化了 {len(self.layout_det_models)} 个版面检测模型")

# 在_try_layout_detection中
logging.info(f"模型 {model_name} 检测到 {seal_count} 个印章")
logging.info(f"模型 {model_name} 耗时: {elapsed_time:.2f}秒")
```

### 6.4 结果对比

可以保存每个模型的检测结果用于对比分析：

```python
def _try_layout_detection_with_comparison(self, doc_preprocessor_images, **kwargs):
    """带对比的检测方法"""
    comparison_results = []
    
    for model_info in self.layout_det_models:
        model = model_info["model"]
        model_name = model_info["name"]
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行检测
        results = list(model(doc_preprocessor_images, **kwargs))
        
        # 记录耗时
        elapsed = time.time() - start_time
        
        # 统计印章数量
        seal_count = sum(
            sum(1 for box in res.get("boxes", []) if box["label"].lower() == "seal")
            for res in results
        )
        
        # 保存对比信息
        comparison_results.append({
            "model_name": model_name,
            "seal_count": seal_count,
            "elapsed_time": elapsed,
            "results": results
        })
        
        # 如果找到印章就返回
        if seal_count > 0:
            return results, model_name, comparison_results
    
    # 返回最后一个模型的结果
    return comparison_results[-1]["results"], comparison_results[-1]["model_name"], comparison_results
```

## 七、常见问题

### 7.1 如何调试？

可以在关键位置添加日志输出：

```python
import logging
logging.basicConfig(level=logging.INFO)

# 在代码中添加日志
logging.info(f"当前使用模型: {model_name}")
logging.info(f"检测结果: {layout_det_results}")
```

### 7.2 模型加载失败？

检查以下几点：
1. 模型名称是否正确
2. 模型文件是否下载完整
3. 设备配置是否正确（GPU/CPU）
4. 内存是否充足

### 7.3 所有模型都未检测到印章？

可能的原因：
1. 阈值设置过高，尝试降低threshold
2. 图像质量问题，检查预处理步骤
3. 印章区域太小或太模糊
4. 模型训练数据与实际场景差异较大

### 7.4 如何评估各模型性能？

可以创建评估脚本：

```python
import time
from paddlex import create_pipeline

pipeline = create_pipeline("custom_seal_recognition", device="gpu:0")

test_images = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = []

for img in test_images:
    start = time.time()
    output = list(pipeline.predict(img))
    elapsed = time.time() - start
    
    for res in output:
        results.append({
            "image": img,
            "model": res.get("used_layout_model", "unknown"),
            "seal_count": len(res.get("seal_res_list", [])),
            "time": elapsed
        })

# 统计分析
import pandas as pd
df = pd.DataFrame(results)
print(df.groupby("model").agg({
    "seal_count": "sum",
    "time": "mean"
}))
```

## 八、最佳实践

### 8.1 开发建议

1. **模块化设计**: 保持代码结构清晰，便于维护
2. **错误处理**: 添加完善的异常处理机制
3. **日志记录**: 记录关键步骤和决策信息
4. **性能监控**: 监控各模型的耗时和成功率
5. **版本管理**: 使用Git管理代码变更

### 8.2 部署建议

1. **模型缓存**: 预加载模型以减少首次推理时间
2. **批处理**: 对于大量图像，使用批处理提高效率
3. **资源管理**: 合理分配GPU资源，避免内存溢出
4. **监控告警**: 部署监控系统，及时发现问题

### 8.3 优化方向

1. **模型剪枝**: 对模型进行剪枝以提高速度
2. **量化部署**: 使用INT8量化减少计算量
3. **模型蒸馏**: 将大模型知识蒸馏到小模型
4. **硬件加速**: 使用TensorRT等加速引擎

## 九、总结

通过本文档，您应该能够：

1. ✅ 理解PaddleX Pipeline的基本架构
2. ✅ 创建支持多模型级联的自定义Pipeline
3. ✅ 配置和使用自定义Pipeline
4. ✅ 根据实际需求进行定制和优化

自定义Pipeline的核心优势：

- **更高的鲁棒性**: 多模型级联提高检测成功率
- **灵活的配置**: 可以根据场景调整模型组合
- **易于扩展**: 可以方便地添加新的模型和逻辑
- **完整的生态**: 复用PaddleX的完整功能

如有问题或建议，欢迎反馈！

---

**文档版本**: v1.0  
**更新日期**: 2025-10-20  
**作者**: wubin

