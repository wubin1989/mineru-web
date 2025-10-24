#!/usr/bin/env python3
"""
测试印章角度计算功能的脚本
"""
import json
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from seal_cascade import calculate_seal_angle


def test_seal_angle_calculation():
    """测试印章角度计算功能"""
    
    print("=" * 80)
    print("测试印章角度计算功能")
    print("=" * 80)
    
    # 测试用例1: 正常矩形（无旋转）
    print("\n测试用例 1: 正常矩形（无旋转）")
    coordinate1 = [100, 100, 200, 150]
    angle_info1 = calculate_seal_angle(coordinate1)
    print(f"坐标: {coordinate1}")
    print(f"角度信息: {json.dumps(angle_info1, indent=2, ensure_ascii=False)}")
    
    # 测试用例2: 正方形印章
    print("\n测试用例 2: 正方形印章")
    coordinate2 = [100, 100, 200, 200]
    angle_info2 = calculate_seal_angle(coordinate2)
    print(f"坐标: {coordinate2}")
    print(f"角度信息: {json.dumps(angle_info2, indent=2, ensure_ascii=False)}")
    
    # 测试用例3: 竖向矩形
    print("\n测试用例 3: 竖向矩形")
    coordinate3 = [100, 100, 150, 200]
    angle_info3 = calculate_seal_angle(coordinate3)
    print(f"坐标: {coordinate3}")
    print(f"角度信息: {json.dumps(angle_info3, indent=2, ensure_ascii=False)}")
    
    # 测试用例4: 从实际JSON文件中读取
    print("\n测试用例 4: 从实际JSON文件中读取")
    json_file = Path(__file__).parent / "output" / "test2" / "page_0_res.json"
    
    if json_file.exists():
        print(f"读取文件: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        boxes = data.get("layout_det_res", {}).get("boxes", [])
        print(f"找到 {len(boxes)} 个检测框")
        
        for i, box in enumerate(boxes):
            label = box.get("label", "")
            if label == "seal":
                print(f"\n印章 {i+1}:")
                coordinate = box["coordinate"]
                score = box["score"]
                
                print(f"  原始坐标: {coordinate}")
                print(f"  置信度: {score:.4f}")
                
                # 计算角度
                angle_info = calculate_seal_angle(coordinate)
                print(f"  角度信息:")
                for key, value in angle_info.items():
                    if key == "error":
                        print(f"    ⚠️  错误: {value}")
                    else:
                        print(f"    - {key}: {value}")
    else:
        print(f"⚠️  文件不存在: {json_file}")
        print("跳过实际数据测试")
    
    print("\n" + "=" * 80)
    print("✓ 测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_seal_angle_calculation()

