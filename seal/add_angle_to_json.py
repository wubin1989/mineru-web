#!/usr/bin/env python3
"""
为现有的印章识别结果JSON文件添加角度信息
"""
import json
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from seal_cascade import calculate_seal_angle


def process_json_file(json_path: Path, output_path: Path = None):
    """
    处理JSON文件，为每个印章添加角度信息
    
    Args:
        json_path: 输入JSON文件路径
        output_path: 输出JSON文件路径（如果为None，则覆盖原文件）
    """
    print(f"\n处理文件: {json_path}")
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取所有检测框
    boxes = data.get("layout_det_res", {}).get("boxes", [])
    print(f"找到 {len(boxes)} 个检测框")
    
    # 统计印章数量
    seal_count = 0
    updated_count = 0
    
    # 为每个印章添加角度信息
    for i, box in enumerate(boxes):
        label = box.get("label", "")
        
        if label == "seal":
            seal_count += 1
            coordinate = box["coordinate"]
            
            # 如果已经有角度信息，跳过
            if "angle_info" in box:
                print(f"  印章 {seal_count}: 已有角度信息，跳过")
                continue
            
            # 计算角度
            angle_info = calculate_seal_angle(coordinate)
            box["angle_info"] = angle_info
            updated_count += 1
            
            print(f"  印章 {seal_count}:")
            print(f"    - 坐标: {coordinate}")
            print(f"    - 旋转角度: {angle_info.get('rotation_angle', 0):.2f}°")
            print(f"    - 顺时针角度: {angle_info.get('clockwise_angle', 0):.2f}°")
            print(f"    - 是否旋转: {'是' if angle_info.get('is_rotated', False) else '否'}")
    
    # 保存更新后的JSON
    if output_path is None:
        output_path = json_path
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"\n✓ 完成!")
    print(f"  - 总印章数: {seal_count}")
    print(f"  - 更新数量: {updated_count}")
    print(f"  - 保存到: {output_path}")
    
    return seal_count, updated_count


def main():
    """主函数"""
    print("=" * 80)
    print("为印章识别结果添加角度信息")
    print("=" * 80)
    
    # 查找所有需要处理的JSON文件
    base_dir = Path(__file__).parent / "output"
    
    if not base_dir.exists():
        print(f"错误: 输出目录不存在: {base_dir}")
        return
    
    # 查找所有的 *_res.json 文件
    json_files = list(base_dir.glob("**/*_res.json"))
    
    if not json_files:
        print(f"未找到任何结果JSON文件在: {base_dir}")
        return
    
    print(f"\n找到 {len(json_files)} 个JSON文件:")
    for i, json_file in enumerate(json_files, 1):
        print(f"  {i}. {json_file.relative_to(base_dir)}")
    
    # 处理每个文件
    total_seals = 0
    total_updated = 0
    
    for json_file in json_files:
        try:
            seal_count, updated_count = process_json_file(json_file)
            total_seals += seal_count
            total_updated += updated_count
        except Exception as e:
            print(f"\n✗ 处理文件失败: {json_file}")
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("📊 总体统计")
    print("=" * 80)
    print(f"  处理文件数: {len(json_files)}")
    print(f"  总印章数: {total_seals}")
    print(f"  更新数量: {total_updated}")
    print("=" * 80)


if __name__ == "__main__":
    main()

