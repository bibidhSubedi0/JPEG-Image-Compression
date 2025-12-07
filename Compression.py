from PIL import Image
import numpy as np
import math
from scipy.fftpack import dct
from scipy.fftpack import idct
import matplotlib.pyplot as plt
from skimage.transform import resize




Quantization1 = [
 [1, 1, 1, 1, 2, 2, 3, 3],
 [1, 1, 1, 2, 2, 3, 3, 3],
 [1, 1, 2, 2, 3, 3, 3, 3],
 [1, 2, 2, 3, 3, 4, 4, 3],
 [2, 2, 3, 3, 3, 4, 4, 4],
 [2, 3, 3, 4, 4, 4, 4, 4],
 [3, 3, 3, 4, 4, 4, 4, 4],
 [3, 3, 3, 3, 4, 4, 4, 4]
]

Quantization2 = \
[[1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1]]




Quantanization3 = [
 [ 8,  8,  8, 16, 24, 40, 56, 64],
 [ 8,  8, 16, 16, 32, 56, 64, 56],
 [ 8, 16, 16, 24, 40, 64, 72, 56],
 [16, 16, 24, 32, 56, 176, 248, 72],
 [24, 32, 40, 56, 72, 200, 192, 256],
 [40, 56, 64, 256, 184, 192, 208, 176],
 [56, 64, 72, 256, 192, 208, 208, 184],
 [64, 56, 56, 72, 256, 176, 184, 184]
]

Quantization_table = {1:Quantization1,2:Quantization2,3:Quantization2}

class wrangling:
    #Step:1 Color Space Conversion and Downsampling
    def ColorSpaceConversion(self, img):
        # Luminance
        Y   = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        # Blue
        Cb  = -0.1687 * img[:,:,0] - 0.3313 * img[:,:,1] + 0.5 * img[:,:,2] + 128
        # Red
        Cr  = 0.5 * img[:,:,0] - 0.4187 * img[:,:,1] -0.0813 * img[:,:,2] + 128
        return Y,Cb,Cr

    #Input Argument must be a 2D np array
    def Downsampling(self, img):
        x,y = img.shape[:2]
        downsampled = np.zeros((int(x/2),int(y/2)))

        for i in range (0,x-x %2,2):
            for j in range (0,y-y%2,2):
                downsampled[i//2,j//2] = (img[i+0,j+0] + img[i+0,j+1]+ img[i+1,j+0]+ img[i+1,j+1]) / 4 
        return downsampled
    
    
    
    def zigzag_traversal(self, matrix):
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

    
    def pack0ToList(self, res):
        """
        Pack consecutive zeros into [0, count] markers.
        Encoding scheme:
        - For non-zero values, append the value.
        - For a run of k zeros (k >= 1), append [0, k].
        This is applied to the whole flattened list.
        """
        ans = []
        i = 0
        n = len(res)
        while i < n:
            if res[i] == 0:
                # count consecutive zeros starting at i
                j = i
                while j < n and res[j] == 0:
                    j += 1
                count = j - i
                ans.extend([0, count])
                i = j
            else:
                ans.append(res[i])
                i += 1
        return ans

  
    def blockDCT(self, block):
        block = block.astype(float) - 128
        return dct(dct(block.T, norm='ortho').T, norm='ortho')


    def PaddingAndDCT(self,padded,element):
        DCT_padded = np.zeros_like(padded)
        for i in range(0, padded.shape[0], 8):
            for j in range(0, padded.shape[1], 8):
                DCT_padded[i:i+8,j:j+8] = Op.blockDCT(padded[i:i+8,j:j+8]) // Quantization_table[element]
        return DCT_padded

    def pad(self, to_pad,x,y):
        pad_x = (8 - x % 8) % 8
        pad_y = (8 - y % 8) % 8
        return np.pad(to_pad, ((0,pad_x),(0,pad_y)), mode='constant', constant_values=0),x+pad_x,y+pad_y

    def blockify(self,mat):
        traversed8_8 = np.zeros(64,dtype=float)
        ans = []
        tem = []
        for i in range(0, mat.shape[0], 8):
            for j in range(0, mat.shape[1], 8):
                # print(mat[i:i+8,j:j+8])
                traversed8_8 = self.zigzag_traversal(mat[i:i+8,j:j+8])
                tem.extend(traversed8_8)
                ans.extend(self.pack0ToList(traversed8_8))
        return ans,tem



#Image as input
image = Image.open("demo2.png")
image_rgb = image.convert("RGB")
image_array = np.array(image_rgb)


Op = wrangling()
Y,Cb,Cr = Op.ColorSpaceConversion(image_array)
Downsampled_Cb = Op.Downsampling(Cb)
Downsampled_Cr = Op.Downsampling(Cr)


#Step:2 DCT and quantanization 

component = {1: Y, 2: Downsampled_Cb, 3: Downsampled_Cr}

saved_data = {}  # dictionary to hold DCT coefficients and sizes
CompressionData = {}
for element in range(1, 4):
    target = component[element]
    OriginalRow, OriginalCol = target.shape[0:2]

    # Padding and DCT
    target_padded, PaddedRow, PaddedCol = Op.pad(target, OriginalRow, OriginalCol)
    DCT_padded = Op.PaddingAndDCT(target_padded, element)

    # Blockify
    imgdata, _ = Op.blockify(DCT_padded)
    imgdata = np.array(imgdata, dtype=float)

    # Store relevant info in dictionary
    saved_data[f"{element}_imgdata"] = imgdata
    saved_data[f"{element}_PaddedRow"] = PaddedRow
    saved_data[f"{element}_PaddedCol"] = PaddedCol

    #Compute Compression Ratio
    
    CompressionData[element] = imgdata.size/component[element].size

# Compute average compression ratio
avg_compression = sum(CompressionData.values()) / len(CompressionData)
print(f"Average compression ratio: {avg_compression:.4f}")



