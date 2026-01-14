import cv2
import numpy as np
import math

def add_signature(img):
    sign = cv2.imread("Sign.bmp")   
    h = len(img)        
    w = len(img[0])    
    sh = len(sign)      
    sw = len(sign[0])       
    for y in range(sh):
        for x in range(sw):
            b, g, r = sign[y, x]    
            if not (b < 15 and g < 15 and r < 15):     
                img[h - sh + y, w - sw + x] = [b, g, r]     
    return img

img = cv2.imread("Picture.bmp")  
print("1. 彩色轉灰階")
print("2. 最近鄰縮放")
print("3. 線性縮放")
print("4. 旋轉")
print(" ")

choice = int(input("請選擇功能 (1~4): "))

if choice == 1:
    h = len(img)        
    w = len(img[0])     
    c = len(img[0][0])  
    gray = np.zeros((h, w, c), dtype=np.uint8)  
    for y in range(h):
        for x in range(w):
            b, g, r = img[y, x]                         
            value = int(0.3 * r + 0.6 * g + 0.1 * b)    
            gray[y, x] = [value, value, value]         
    gray = add_signature(gray)              
    cv2.imwrite("output_gray.bmp", gray)    
    print("已完成")

elif choice == 2:
    scale = float(input("輸入縮放比例 (0.1~2): "))
    h = len(img)       
    w = len(img[0])     
    c = len(img[0][0])  
    new_h = int(h * scale)  
    new_w = int(w * scale)  

    forward_img = np.zeros((new_h, new_w, c), dtype=np.uint8)  
    for y in range(h):
        for x in range(w):
            x_newf = int(round(x * scale))  
            y_newf = int(round(y * scale))   
            if 0 <= x_newf < new_w and 0 <= y_newf < new_h:   
                forward_img[y_newf, x_newf] = img[y, x]
    forward_img = add_signature(forward_img)    
    cv2.imwrite("output_nearest_forward.bmp", forward_img)  
    print("Forward Mapping 已完成")

    inverse_img = np.zeros((new_h, new_w, c), dtype=np.uint8)   
    for y in range(new_h):
        for x in range(new_w):  
            x_newi = int(round(x / scale))   
            y_newi = int(round(y / scale))   
            if x_newi >= w:      
                x_newi = w - 1   
            if y_newi >= h:      
                y_newi = h - 1   
            inverse_img[y, x] = img[y_newi, x_newi]   
    inverse_img = add_signature(inverse_img)    
    cv2.imwrite("output_nearest_inverse.bmp", inverse_img)  
    print("Inverse Mapping 已完成")

elif choice == 3:
    scale = float(input("輸入縮放比例 (0.1~2): "))
    h = len(img)        
    w = len(img[0])     
    c = len(img[0][0])  
    new_h = int(h * scale)  
    new_w = int(w * scale)  

    forward_img = np.zeros((new_h, new_w, c), dtype=float)   
    count = np.zeros((new_h, new_w), dtype=float)            
    for y in range(h - 1):
        for x in range(w - 1):
            x_new = x * scale       
            y_new = y * scale       
            xint = int(x_new)       
            yint = int(y_new)       
            xflo = x_new - xint     
            yflo = y_new - yint     
            if 0 <= xint < new_w - 1 and 0 <= yint < new_h - 1: 
                A = img[y, x].astype(float)         
                B = img[y, x + 1].astype(float)     
                C = img[y + 1, x].astype(float)     
                D = img[y + 1, x + 1].astype(float) 
                color = (1 - xflo) * (1 - yflo) * A + xflo * (1 - yflo) * B + (1 - xflo) * yflo * C + xflo * yflo * D   
                for dy in range(2):
                    for dx in range(2):
                        nx = xint + dx  
                        ny = yint + dy  
                        if 0 <= nx < new_w and 0 <= ny < new_h:  
                            wx = 1 - abs(dx - xflo)     
                            wy = 1 - abs(dy - yflo)     
                            wght = wx * wy              
                            if wght > 0:                
                                for ch in range(c):      
                                    forward_img[ny, nx, ch] += color[ch] * wght  
                                count[ny, nx] += wght       

    for y in range(new_h):      
        for x in range(new_w):
            if count[y, x] != 0:        
                for ch in range(c):     
                    val = forward_img[y, x, ch] / count[y, x]   

                    if val < 0:
                        val = 0
                    elif val > 255:
                        val = 255
                    forward_img[y, x, ch] = int(round(val))  
    forward_img = forward_img.astype(np.uint8)  
    forward_img = add_signature(forward_img)    
    cv2.imwrite("output_bilinear_forward.bmp", forward_img) 
    print("Forward Mapping 已完成")

    inverse_img = np.zeros((new_h, new_w, c), dtype=np.uint8)   
    for y in range(new_h):
        for x in range(new_w):
            x_newi = x / scale      
            y_newi = y / scale      
            xint = int(x_newi)      
            yint = int(y_newi)      
            xflo = x_newi - xint    
            yflo = y_newi - yint    
            if xint >= w - 1:       
                xint = w - 2        
            if yint >= h - 1:       
                yint = h - 2        
            for ch in range(c):     
                A = img[yint, xint, ch]           
                B = img[yint, xint + 1, ch]       
                C = img[yint + 1, xint, ch]       
                D = img[yint + 1, xint + 1, ch]   
                inverse_img[y, x, ch] = int((1 - xflo) * (1 - yflo) * A + xflo * (1 - yflo) * B + (1 - xflo) * yflo * C + xflo * yflo * D)  
    inverse_img = add_signature(inverse_img)    
    cv2.imwrite("output_bilinear_inverse.bmp", inverse_img) 
    print("Inverse Mapping 已完成")

elif choice == 4:
    angle = float(input("輸入旋轉角度 (0~360): "))
    rad = math.radians(angle)   
    h = len(img)        
    w = len(img[0])     
    c = len(img[0][0])  
    cx = w / 2      
    cy = h / 2      
    corners = np.array([[-cx, -cy], [w - cx - 1, -cy], [-cx, h - cy - 1], [w - cx - 1, h - cy - 1]], dtype=np.float32)  
    rot = np.zeros((4, 2))  
    for i in range(4):
        x, y = corners[i]   
        rot[i, 0] = x * math.cos(rad) - y * math.sin(rad)   
        rot[i, 1] = x * math.sin(rad) + y * math.cos(rad)   
    min_x = np.min(rot[:, 0])   
    min_y = np.min(rot[:, 1])   
    max_x = np.max(rot[:, 0])   
    max_y = np.max(rot[:, 1])   
    new_w = int(math.ceil(max_x - min_x))   
    new_h = int(math.ceil(max_y - min_y))   
    
    forward_img = np.zeros((new_h, new_w, c), dtype=np.uint8)   
    for y in range(h):
        for x in range(w):
            x_shift = x - cx    
            y_shift = y - cy    
            x_newf = int(round(x_shift * math.cos(rad) - y_shift * math.sin(rad) - min_x))   
            y_newf = int(round(x_shift * math.sin(rad) + y_shift * math.cos(rad) - min_y))   
            if 0 <= x_newf < new_w and 0 <= y_newf < new_h:   
                forward_img[y_newf, x_newf] = img[y, x]
    forward_img = add_signature(forward_img)    
    cv2.imwrite("output_rotate_forward.bmp", forward_img)   
    print("Forward Mapping 已完成")

    inverse_img = np.zeros((new_h, new_w, c), dtype=np.uint8)   
    for y in range(new_h):
        for x in range(new_w):
            x_shift = x + min_x     
            y_shift = y + min_y     
            x_newi = int(round(x_shift * math.cos(-rad) - y_shift * math.sin(-rad) + cx))     
            y_newi = int(round(x_shift * math.sin(-rad) + y_shift * math.cos(-rad) + cy))     
            if 0 <= x_newi < w and 0 <= y_newi < h:   
                inverse_img[y, x] = img[y_newi, x_newi] 
    inverse_img = add_signature(inverse_img)   
    cv2.imwrite("output_rotate_inverse.bmp", inverse_img)   
    print("Inverse Mapping 已完成")
else:
    print("無效的選項")
