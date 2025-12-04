from PIL import Image
import numpy as np

#Image as input
image = Image.open("trial.png")
image_rgb = image.convert("RGB")
image_array = np.array(image_rgb)

def display():
    print(image_rgb.size)
    print(image_rgb.mode)


#Step:1 Color Space Conversion and Downsampling
def ColorSpaceConversion(img):
    Y   = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    Cb  = -0.1687 * img[:,:,0] - 0.3313 * img[:,:,1] + 0.5 * img[:,:,2] + 128
    Cr  = 0.5 * img[:,:,0] - 0.4187 * img[:,:,1] -0.0813 * img[:,:,2] + 128
    return Y,Cb,Cr

#Input Argument must be a 2D np array
def Downsampling(img):
    x,y = img.shape[:2]
   
    downsampled = np.zeros((int(x/2),int(y/2)))
    k = 0
    l = 0
    for i in range (0,x-3,2):
        k = k+1
        l = 0
        for j in range (0,y-2,2):
            downsampled[k,l] = (img[i+0,j+0] + img[i+0,j+1]+ img[i+1,j+0]+ img[i+1,j+1]) / 4 
            # print("test: ",downsampled[k,l] ," k: " ,k ," l: ", l," i = ",i," j: ",j)
            l = l + 1
 
    return downsampled
    


Y,Cb,Cr = ColorSpaceConversion(image_array)
Downsampled_Cb = Downsampling(Cb)
Downsampled_Cr = Downsampling(Cr)

image = Image.fromarray(Downsampled_Cb)
image.show()
print(Y[0:2,4:6])
