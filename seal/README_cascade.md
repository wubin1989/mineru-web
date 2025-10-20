# 多模型级联印章识别工具使用说明

> **作者**: wubin  
> **日期**: 2025-10-20

## 概述

`seal_cascade.py` 是基于 `seal.py` 开发的多模型级联印章识别工具。它在版面检测（LayoutDetection）阶段会依次尝试多个模型，直到检测到印章为止。

## 核心特性

✅ **多模型级联检测**: 依次尝试多个模型，提高印章检测成功率  
✅ **智能切换**: 检测到印章后立即停止，避免不必要的计算  
✅ **模型统计**: 自动统计各模型的使用情况  
✅ **完全兼容**: 与原有 `seal.py` 的其他功能完全兼容

## 模型级联顺序

1. **PP-DocLayout-L**: 通用版面检测模型，速度快
2. **PP-DocLayout_plus-L**: 增强版版面检测模型，精度更高  
3. **RT-DETR-H_layout_3cls**: 高精度检测模型，计算量较大

如果第一个模型检测到印章，就不会使用后续模型。只有在未检测到印章时，才会尝试下一个模型。

## 文件说明

### 1. CustomSealRecognition.yaml

多模型级联配置文件，包含：
- 三个版面检测模型的配置（按优先级排列）
- 文档预处理配置（方向分类、文档矫正）
- 印章OCR识别配置

### 2. seal_cascade.py

多模型级联印章识别脚本，包含：
- 自定义Pipeline实现类 `_CustomSealRecognitionPipeline`
- 多模型级联检测逻辑
- 完整的PDF/图片处理流程
- 模型使用统计功能

## 使用方法

### 基本用法

```bash
# 处理单个PDF文件
python seal_cascade.py --input 招股书.pdf --output ./output

# 处理单个图片文件
python seal_cascade.py --input seal.jpg --output ./output

# 处理整个文件夹
python seal_cascade.py --input ./test_images --output ./output
```

### 高级参数

```bash
python seal_cascade.py \
  --input 招股书.pdf \
  --output ./output \
  --config CustomSealRecognition.yaml \
  --max-workers 4 \
  --zoom 2.0 \
  --use-orientation \
  --use-unwarping
```

### 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--input` | 输入PDF/图片文件或文件夹路径 | 招股书.pdf |
| `--output` | 输出目录路径 | ./output |
| `--config` | 自定义配置文件路径 | CustomSealRecognition.yaml |
| `--zoom` | PDF转图片的缩放倍数 | 2.0 |
| `--max-workers` | 并发处理的最大线程数 | 4 |
| `--use-orientation` | 是否使用文档方向分类 | False |
| `--use-unwarping` | 是否使用文档矫正 | False |

## 输出结果

处理完成后，会在输出目录生成以下文件：

```
output/
├── 文件名/
│   ├── temp_images/           # 临时图片文件
│   │   ├── page_0.png
│   │   ├── page_1.png
│   │   └── ...
│   ├── page_0_res.json        # JSON结果文件
│   ├── page_0_layout.png      # 版面检测可视化
│   ├── seal_cropped_page0_1_score0.950.png  # 裁剪的印章图片
│   └── ...
```

## 示例

### 示例1：处理单个PDF

```bash
python seal_cascade.py \
  --input testdata/招股书.pdf \
  --output ./output \
  --max-workers 4
```

### 示例2：处理图片文件夹（启用预处理）

```bash
python seal_cascade.py \
  --input testdata/seals1/ \
  --output ./output \
  --use-orientation \
  --use-unwarping \
  --max-workers 8
```

### 示例3：使用自定义配置

```bash
python seal_cascade.py \
  --input document.pdf \
  --output ./output \
  --config MyCustomConfig.yaml
```

## 输出信息解读

### 模型级联日志

在运行过程中，会看到类似以下的日志：

```
尝试使用模型 PP-DocLayout-L 进行版面检测...
✗ 模型 PP-DocLayout-L 未检测到印章，尝试下一个模型...

尝试使用模型 PP-DocLayout_plus-L 进行版面检测...
✓ 模型 PP-DocLayout_plus-L 成功检测到印章！
```

### 模型使用统计

处理完成后会显示模型使用统计：

```
📊 模型使用统计:
  - PP-DocLayout-L: 5 次
  - PP-DocLayout_plus-L: 3 次
  - RT-DETR-H_layout_3cls: 1 次
```

这表明：
- 5个页面使用第一个模型就检测到印章
- 3个页面需要使用第二个模型
- 1个页面需要使用第三个模型

## 自定义配置

### 修改模型顺序

编辑 `CustomSealRecognition.yaml`，调整 `LayoutDetectionModels` 列表中模型的顺序：

```yaml
SubModules:
  LayoutDetectionModels:
    # 将你最常用的模型放在前面
    - model_name: PP-DocLayout_plus-L
      # ... 其他配置
    
    - model_name: PP-DocLayout-L
      # ... 其他配置
```

### 调整检测阈值

降低阈值可以提高召回率（检测到更多印章），但可能增加误检：

```yaml
SubModules:
  LayoutDetectionModels:
    - model_name: PP-DocLayout-L
      threshold: 0.3  # 降低到0.3
```

### 添加更多模型

在配置文件中添加更多模型：

```yaml
SubModules:
  LayoutDetectionModels:
    - model_name: PP-DocLayout-L
      # ...
    
    - model_name: PP-DocLayout_plus-L
      # ...
    
    - model_name: RT-DETR-H_layout_3cls
      # ...
    
    - model_name: YourCustomModel
      # ...
```

## 性能对比

与原始 `seal.py` 相比：

| 指标 | seal.py | seal_cascade.py |
|-----|---------|----------------|
| 印章检测成功率 | 单一模型 | 多模型级联，更高 |
| 处理速度 | 固定 | 智能切换，平均稍慢 |
| 资源消耗 | 低 | 中等（按需加载） |
| 适用场景 | 印章清晰 | 各种复杂场景 |

## 常见问题

### Q1: 所有模型都未检测到印章怎么办？

**A**: 尝试以下方法：
1. 降低配置文件中的 `threshold` 值
2. 启用文档预处理: `--use-orientation --use-unwarping`
3. 提高PDF转图片的分辨率: `--zoom 3.0`
4. 检查图片质量，确保印章清晰可见

### Q2: 处理速度太慢？

**A**: 优化建议：
1. 减少 `--max-workers` 参数（避免过度并发）
2. 将成功率高的模型放在配置文件前面
3. 关闭不必要的预处理选项
4. 使用GPU加速（如果可用）

### Q3: 如何调试模型选择？

**A**: 查看日志输出：
- 每个模型的尝试都会有日志
- 成功的模型会显示 ✓ 标记
- 最终会统计各模型的使用次数

### Q4: 可以只使用某个特定模型吗？

**A**: 可以，修改配置文件，只保留一个模型：

```yaml
SubModules:
  LayoutDetectionModels:
    - model_name: PP-DocLayout_plus-L
      # ... 只保留这一个
```

### Q5: 内存占用过高？

**A**: 
1. 减少 `--max-workers` 参数
2. 减少配置中的模型数量
3. 一次处理较少的文件

## 技术细节

### Pipeline架构

```
seal_cascade.py
├── _CustomSealRecognitionPipeline
│   ├── __init__(): 初始化多个模型
│   ├── _has_seal_boxes(): 检查是否检测到印章
│   ├── _try_layout_detection(): 级联检测逻辑
│   └── predict(): 完整预测流程
└── 处理流程
    ├── 第一步：PDF转图片/加载图片
    ├── 第二步：多模型级联识别
    └── 第三步：裁剪印章区域
```

### 模型级联逻辑

```python
for model in models:
    results = model.detect(image)
    if has_seal(results):
        return results, model_name
    # 否则继续尝试下一个模型
return last_results, last_model_name
```

## 最佳实践

### 1. 开发环境

```bash
# 首先在小数据集上测试
python seal_cascade.py --input test.pdf --output ./test_output

# 确认无误后处理大批量数据
python seal_cascade.py --input ./large_dataset --output ./output --max-workers 8
```

### 2. 生产环境

- 使用固定的配置文件版本
- 定期检查模型使用统计，优化模型顺序
- 监控处理时间和成功率
- 对失败的案例单独处理

### 3. 性能优化

- 将最常用、成功率最高的模型放在前面
- 根据实际场景调整阈值
- 合理设置并发线程数
- 使用GPU加速（如果可用）

## 与原版seal.py的区别

| 特性 | seal.py | seal_cascade.py |
|-----|---------|----------------|
| 版面检测模型 | 单一模型 | 多模型级联 |
| 配置文件 | SealRecognition.yaml | CustomSealRecognition.yaml |
| Pipeline类 | 使用PaddleX原生 | 自定义实现 |
| 模型统计 | 无 | 有 |
| 参数 | 无--config参数 | 支持--config参数 |
| 适用场景 | 标准场景 | 复杂场景 |

## 贡献与反馈

如有问题或建议，欢迎反馈！

---

**版本**: v1.0  
**最后更新**: 2025-10-20  
**作者**: wubin

