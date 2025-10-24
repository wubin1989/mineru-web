#!/usr/bin/env python3
"""
测试基于OCR的印章角度检测功能
"""
import json
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from seal_cascade import calculate_text_orientation_angle, calculate_seal_angle_with_text


def test_text_orientation():
    """测试文本方向计算"""
    print("=" * 80)
    print("测试文本方向角度计算")
    print("=" * 80)
    
    # 测试用例1: 全部正常
    print("\n测试用例 1: 全部正常方向")
    rec_texts = ["文字1", "文字2", "文字3"]
    textline_angles = [0, 0, 0]
    angle, stats = calculate_text_orientation_angle(rec_texts, textline_angles)
    print(f"  文本: {rec_texts}")
    print(f"  方向: {textline_angles}")
    print(f"  纠正角度: {angle}°")
    print(f"  统计: {stats}")
    assert angle == 0, "全部正常应该不需要纠正"
    assert stats['confidence'] == 1.0, "置信度应该是1.0"
    
    # 测试用例2: 全部倒置
    print("\n测试用例 2: 全部倒置")
    rec_texts = ["文字1", "文字2", "文字3"]
    textline_angles = [1, 1, 1]
    angle, stats = calculate_text_orientation_angle(rec_texts, textline_angles)
    print(f"  文本: {rec_texts}")
    print(f"  方向: {textline_angles}")
    print(f"  纠正角度: {angle}°")
    print(f"  统计: {stats}")
    assert angle == 180, "全部倒置应该旋转180度"
    assert stats['confidence'] == 1.0, "置信度应该是1.0"
    
    # 测试用例3: 混合（多数倒置）
    print("\n测试用例 3: 混合（多数倒置）")
    rec_texts = ["文字1", "文字2", "文字3", "文字4", "文字5"]
    textline_angles = [0, 1, 1, 1, 1]  # 1个正常，4个倒置
    angle, stats = calculate_text_orientation_angle(rec_texts, textline_angles)
    print(f"  文本: {rec_texts}")
    print(f"  方向: {textline_angles}")
    print(f"  纠正角度: {angle}°")
    print(f"  统计: {stats}")
    assert angle == 180, "多数倒置应该旋转180度"
    assert stats['confidence'] == 0.8, "置信度应该是0.8"
    
    # 测试用例4: 混合（多数正常）
    print("\n测试用例 4: 混合（多数正常）")
    rec_texts = ["文字1", "文字2", "文字3", "文字4", "文字5"]
    textline_angles = [0, 0, 0, 1, 1]  # 3个正常，2个倒置
    angle, stats = calculate_text_orientation_angle(rec_texts, textline_angles)
    print(f"  文本: {rec_texts}")
    print(f"  方向: {textline_angles}")
    print(f"  纠正角度: {angle}°")
    print(f"  统计: {stats}")
    assert angle == 0, "多数正常应该不需要纠正"
    assert stats['confidence'] == 0.6, "置信度应该是0.6"
    
    # 测试用例5: 有空字符串
    print("\n测试用例 5: 包含空字符串")
    rec_texts = ["文字1", "", "文字3", "", "文字5"]
    textline_angles = [1, 0, 1, 0, 1]  # 空字符串应该被过滤
    angle, stats = calculate_text_orientation_angle(rec_texts, textline_angles)
    print(f"  文本: {rec_texts}")
    print(f"  方向: {textline_angles}")
    print(f"  纠正角度: {angle}°")
    print(f"  统计: {stats}")
    print(f"  有效文本数: {stats['total']}")
    assert stats['total'] == 3, "应该只统计有效文本"
    assert angle == 180, "3个倒置应该需要纠正"
    
    print("\n✓ 所有测试通过!")


def test_with_real_data():
    """使用实际JSON数据测试"""
    print("\n" + "=" * 80)
    print("测试实际JSON数据")
    print("=" * 80)
    
    json_file = Path(__file__).parent / "output" / "test2" / "page_0_res.json"
    
    if not json_file.exists():
        print(f"⚠️  文件不存在: {json_file}")
        print("跳过实际数据测试")
        return
    
    print(f"\n读取文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    boxes = data.get("layout_det_res", {}).get("boxes", [])
    seal_res_list = data.get("seal_res_list", [])
    
    print(f"找到 {len(boxes)} 个检测框")
    print(f"找到 {len(seal_res_list)} 个印章OCR结果")
    
    for i, (box, seal_res) in enumerate(zip(boxes, seal_res_list)):
        if box.get("label") == "seal":
            print(f"\n印章 {i+1}:")
            coordinate = box["coordinate"]
            rec_texts = seal_res.get("rec_texts", [])
            textline_angles = seal_res.get("textline_orientation_angles", [])
            
            print(f"  坐标: {coordinate}")
            print(f"  文本数: {len(rec_texts)}")
            print(f"  文本内容: {[t for t in rec_texts if t]}")  # 只显示非空文本
            print(f"  方向列表: {textline_angles}")
            
            # 计算角度
            angle_info = calculate_seal_angle_with_text(
                coordinate=coordinate,
                rec_texts=rec_texts,
                textline_orientation_angles=textline_angles
            )
            
            print(f"  角度信息:")
            for key, value in angle_info.items():
                print(f"    {key}: {value}")


def main():
    """主函数"""
    try:
        test_text_orientation()
        test_with_real_data()
        
        print("\n" + "=" * 80)
        print("✓ 所有测试完成!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

