from paddleocr import SealRecognition

pipeline = SealRecognition()
pipeline.export_paddlex_config_to_yaml("SealRecognition.yaml")