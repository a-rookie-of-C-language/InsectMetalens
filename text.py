import numpy as np


# 假设这是你的 random_small_phase 函数
def random_small_phase(shape):
    return np.random.uniform(-0.1 * np.pi, 0.1 * np.pi, shape)


# 检查随机数生成器的种子是否被设置
print("Before setting seed:")
print(random_small_phase((2, 2)))

# 假设在这里设置了种子
np.random.seed(0)

print("\nAfter setting seed:")
print(random_small_phase((2, 2)) == random_small_phase((2, 2)))
# print(random_small_phase((2, 2)))
# print(random_small_phase((2, 2)))  # 这将产生与上面相同的随机数
