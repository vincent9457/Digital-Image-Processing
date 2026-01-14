import cv2
import numpy as np
import time

img = cv2.imread("Picture.jpg")         
x, y, w, h = map(int, input().split())  
start = time.time()

roi_str = ""    
for j in range(h):
    for i in range(w):
        pixel = img[y+j, x+i]                                    
        b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])    
        roi_str += f"{b},{g},{r};"                               

values = roi_str.split(";")[:-1]                        
pixels = [list(map(int, v.split(","))) for v in values] 
roi = np.zeros((h, w, 3), dtype=np.uint8)               
index = 0
for j in range(h):
    for i in range(w):
        roi[j, i, 0] = pixels[index][0]  
        roi[j, i, 1] = pixels[index][1]  
        roi[j, i, 2] = pixels[index][2]  
        index += 1

cv2.imwrite("NewPicture.jpg", roi)  
end = time.time()
print("逐點操作耗時:", end - start, "秒")