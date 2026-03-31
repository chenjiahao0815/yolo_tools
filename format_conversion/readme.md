# LabelMe JSON 批量转 Mask 工具

## 功能简介
批量将 [LabelMe](https://github.com/wkentaro/labelme) 标注的 JSON 文件转换为二值掩码图像（`.png`）。  
支持 **多边形（polygon）** 和 **矩形（rectangle）** 两种标注形状。  
生成的掩码中，标注区域为白色（像素值 1），背景为黑色（像素值 0），保存为单通道 8 位 PNG 图像。

## 函数原型
```python
batch_json_to_mask(input_dir: str, output_dir: str, target_class: Union[str, int, None] = None) -> None
