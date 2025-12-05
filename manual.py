# #This script contains the manual calculation of the DCT coeffiencients of 8*8 pixel block
import numpy as np
# import math 

# def computeDCT(block8_8): 
#     block8_8 = block8_8.astype(float) - 128 #Shifted so that the pixel value is centered aroud 0 
#     DCT_8_8 = np.zeros((8,8)) #Initialize the Coeffiient matrix for 8*8 cosine frequency block 
#     for u in range (0,8): 
#         for v in range (0,8): 
#             sum_factor = 0 
#             alpha_u = 1 if u > 0 else 1/math.sqrt(2) 
#             alpha_v = 1 if v > 0 else 1/math.sqrt(2) 
#             for x in range (0,8): 
#                 for y in range (0,8): 
#                     sum_factor = sum_factor+block8_8[x,y] * np.cos(((2*x+1)*u*np.pi)/16) * np.cos(((2*y+1)*v*np.pi)/16) 
#         DCT_8_8[u,v] = (1/4) * alpha_u * alpha_v * sum_factor 
#         return DCT_8_8


def zigzag_traversal(matrix):
    n = len(matrix)
    result = []

    for d in range(2 * n - 1):
        temp = []

        # starting row for diagonal d
        r = 0 if d < n else d - n + 1
        # starting col for diagonal d
        c = d if d < n else n - 1

        while r < n and c >= 0:
            temp.append(matrix[r][c])
            r += 1
            c -= 1

        # reverse even diagonals (0,2,4,...)
        if d % 2 == 0:
            temp.reverse()

        result.extend(temp)

    return result


def count(res):
    ptr0 = 0
    ptr1 = 1
    ans = []

    while ptr1 < len(res):

        if res[ptr1] != 0:
            ans.append(res[ptr0])
            ptr0 += 1
            ptr1 += 1
            continue

        ans.append(res[ptr0])
        ans.append(0)

        # Move ptr1 forward until non-zero (safe order!)
        while ptr1 < len(res) and res[ptr1] == 0:
            ptr1 += 1

        # Distance of zero streak
        ans.append(ptr1 - ptr0 -1)

        # Move ptr0 to ptr1
        ptr0 = ptr1
        ptr1 = ptr0 + 1

    # Handle last element (if not processed)
    if ptr0 < len(res):
        ans.append(res[ptr0])

    return ans


sample_matrix = \
[[-12, -15,   0,  -1,  -1,  -1,  -1,  -1],
 [  5,  11,   2,  -2,  -1,   0,  -1,  -1],
 [  9,   2,  -2,   0,  -1,  -1,  -1,  -1],
 [  2,  -2,  -2,   0,   0,   0,  -1,  -1],
 [ -1,  -2,  -1,   0,   0,  -1,   0,   0],
 [  0,  -1,  -1,  -1,   0,  -1,   0,   0],
 [ -1,  -1,   0,  -1,   0,   0,   0,   0],
 [ -1,  -1,   0,  -1,   0,   0,   0,   0]]

print("Sample 16x16 Matrix:")
print(sample_matrix)
print("-"*100)
print(zigzag_traversal(sample_matrix))

print("-"*100)
print(count(zigzag_traversal(sample_matrix)))
print("-"*100)
print(f"{len(zigzag_traversal(sample_matrix))} --> {len(count(zigzag_traversal(sample_matrix)))}")