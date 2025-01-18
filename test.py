# 检查像素值的分布
import numpy as np
from PIL import Image
edges = np.array(Image.open("edges.png"))
unique_values, counts = np.unique(edges, return_counts=True)
print("Pixel value distribution:", dict(zip(unique_values, counts)))