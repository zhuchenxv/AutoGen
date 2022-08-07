import numpy as np
import random

print("once-for-all")
num = 2000
# 4860  18*3  *  18*5
son_structure_sample_list = set()
son_structure_sample_list.add((54, 90))
while True:
    a = random.randint(1, 54)
    b = random.randint(1, 90)
    if ((a, b)) not in son_structure_sample_list:
        son_structure_sample_list.add((a, b))
    if (len(son_structure_sample_list) >= num):
        break

son_structure_sample_list = list(son_structure_sample_list)
son_structure_sample_list.sort()
print(son_structure_sample_list[:10])
print(son_structure_sample_list[-10:])
np.save('old/sample_son_structure.npy', son_structure_sample_list)




