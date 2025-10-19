# from paddleocr import SealRecognition

# pipeline = SealRecognition(
#     use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
#     use_doc_unwarping=False, # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
#     # layout_merge_bboxes_mode="union",
#     layout_merge_bboxes_mode="small",
# )
# # ocr = SealRecognition(device="gpu") # 通过 device 指定模型推理时使用 GPU
# output = pipeline.predict("./testdata/1760521757400_银行流水业务专用3.png")
# for res in output:
#     res.print() ## 打印预测的结构化输出
#     res.save_to_img("./output1/")
#     res.save_to_json("./output1/")

# from paddlex import create_pipeline

# pipeline = create_pipeline(pipeline="PP-StructureV3")

# output = pipeline.predict(
#     input="./testdata/seals/3交行文字公章.png",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
#     layout_merge_bboxes_mode="small",
# )
# for res in output:
#     res.print() ## 打印预测的结构化输出
#     res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
#     res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果
#     res.save_to_img(save_path="output") ## 保存当前图像的可视化结果

from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL()
# pipeline = PaddleOCRVL(use_doc_orientation_classify=True) # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
# pipeline = PaddleOCRVL(use_doc_unwarping=True) # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
# pipeline = PaddleOCRVL(use_layout_detection=False) # 通过 use_layout_detection 指定是否使用版面区域检测排序模块
output = pipeline.predict("./testdata/seals/3交行文字公章.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果
    res.save_to_img(save_path="output") ## 保存当前图像的可视化结果