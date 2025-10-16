from paddleocr import PPStructureV3

pipeline = PPStructureV3()
# pipeline = PPStructureV3(lang="en") # 将 lang 参数设置为使用英文文本识别模型。对于其他支持的语言，请参阅第5节：附录部分。默认配置为中英文模型。
# pipeline = PPStructureV3(use_doc_orientation_classify=True) # 通过 use_doc_orientation_classify 指定是否使用文档方向分类模型
# pipeline = PPStructureV3(use_doc_unwarping=True) # 通过 use_doc_unwarping 指定是否使用文本图像矫正模块
# pipeline = PPStructureV3(use_textline_orientation=True) # 通过 use_textline_orientation 指定是否使用文本行方向分类模型
# pipeline = PPStructureV3(device="gpu") # 通过 device 指定模型推理时使用 GPU
output = pipeline.predict("./testdata/应盖未盖.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果