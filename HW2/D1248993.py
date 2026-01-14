from collections import deque
import cv2
import numpy as np
import math

def Filter2D(img, kernel):      
    kh, kw = kernel.shape   
    pad_h = kh // 2         
    pad_w = kw // 2         
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode = 'constant')   
    h, w = img.shape     
    output = np.zeros((h, w), dtype = np.float32)   
    for y in range(h):
        for x in range(w):
            region = padded[y : y + kh, x : x + kw]     
            output[y, x] = np.sum(region * kernel)      
    return output

def gaussian(img, size = 5, sigma = 1.4):   
    k = size // 2         
    kernel = np.zeros((size, size), dtype = np.float32)     
    for i in range(size):
        for j in range(size):
            x = i - k       
            y = j - k       
            kernel[i, j] = math.exp(-(x * x + y * y) / (2 * sigma * sigma)) 
    kernel /= kernel.sum()     
    return Filter2D(img, kernel)    

def my_sobel(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float32)    
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float32)    
    gx = Filter2D(img, sobel_x)     
    gy = Filter2D(img, sobel_y)     
    return gx, gy

def mag_angle(gx, gy):
    mag = np.sqrt(gx ** 2 + gy ** 2)        
    angle = np.rad2deg(np.arctan2(gy, gx))  
    angle[angle < 0] += 180                 
    return mag, angle

def quantize_angle(angle):
    q = np.zeros_like(angle, dtype = np.uint8)  
    q[(angle >= 0) & (angle < 22.5)] = 0        
    q[(angle >= 157.5) & (angle < 180)] = 0     
    q[(angle >= 22.5) & (angle < 67.5)] = 45    
    q[(angle >= 67.5) & (angle < 112.5)] = 90   
    q[(angle >= 112.5) & (angle < 157.5)] = 135 
    return q

def NMS(mag, angle_q):
    h, w = mag.shape    
    nms = np.zeros((h, w), dtype = np.float32)  

    for y in range(1, h-1):
        for x in range(1, w-1):
            direction = angle_q[y, x]    
            m = mag[y, x]               
            if direction == 0:  
                before = mag[y, x-1]    
                after = mag[y, x+1]     
            elif direction == 45:
                before = mag[y-1, x+1]  
                after = mag[y+1, x-1]   
            elif direction == 90:
                before = mag[y-1, x]    
                after = mag[y+1, x]     
            elif direction == 135:
                before = mag[y-1, x-1]  
                after = mag[y+1, x+1]   
            else:
                before = after = 0         
            if m >= before and m >= after:  
                nms[y, x] = m   
            else:
                nms[y, x] = 0
    return nms

def double_threshold(nms):
    high = nms.max() * 0.2  
    low = high * 0.5 
    strong = np.uint8(nms >= high) * 255                
    weak = np.uint8((nms >= low) & (nms < high)) * 255  
    return strong, weak

def hysteresis(strong, weak):
    h, w = strong.shape     
    edges = strong.copy()   
    q = deque()     

    ys, xs = np.where(strong == 255)
    for y, x in zip(ys, xs):
        q.append((y, x))
    
    directions = [(-1,-1), (-1,0), (-1,1), ( 0,-1), ( 0,1), ( 1,-1), ( 1,0), ( 1,1)]
    
    while q:
        y, x = q.popleft()  
        for dy, dx in directions:
            ny = y + dy     
            nx = x + dx     

            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue

            if weak[ny, nx] == 255 and edges[ny, nx] == 0:
                edges[ny, nx] = 255
                q.append((ny, nx))
    return edges

def hough_lines(edge_img, acc_name, threshold = 400, sample_ratio = 0.1):
    h, w = edge_img.shape      
    diag = int(np.hypot(h, w))  
    acc = np.zeros((180, 2*diag), dtype = np.uint32)  
    
    ys, xs = np.where(edge_img > 0) 
    N = len(xs)                  
    if N == 0:                  
        return []
    sample_size = max(1, int(N * sample_ratio))                 
    idx = np.random.choice(N, size = sample_size, replace = False)  
    xs = xs[idx]                  
    ys = ys[idx]                  
    for x, y in zip(xs, ys):
        for theta in range(180):
            rho = x * math.cos(theta * math.pi / 180) + y * math.sin(theta * math.pi / 180) 
            rho = int(rho)      
            rho_idx = rho + diag   
            if 0 <= rho_idx < 2*diag:  
                acc[theta, rho_idx] += 1            
            
            for dtheta in (-1, 0, 1):   
                tp = theta + dtheta     
                if 0 <= tp < 180:   
                    for drho in (-1, 0, 1):
                        rp = rho_idx + drho
                        if 0 <= rp < 2*diag:
                            acc[tp, rp] += 1
    thetas, rhos = np.where(acc > threshold)   
    lines = []  
    for t, r in zip(thetas, rhos):
        lines.append((int(t), int(r - diag)))
    acc_float = acc.astype(np.float32)  
    acc_min = acc_float.min()   
    acc_max = acc_float.max()   
    if acc_max - acc_min == 0:  
        acc_norm = np.zeros_like(acc_float)     
    else:
        acc_norm = (acc_float - acc_min) / (acc_max - acc_min) * 255   
    acc_img = acc_norm.astype(np.uint8) 
    acc_img = acc_img.T     
    acc_img = cv2.resize(acc_img, (800, 800))  
    acc_img = cv2.cvtColor(acc_img, cv2.COLOR_GRAY2BGR)
    acc_img = add_signature(acc_img, "Sign.bmp")
    cv2.imwrite(acc_name, acc_img)
    return lines

def draw_hough_lines(img, lines):
    h, w = img.shape[:2]    
    out = img.copy()    
    for theta, rho in lines:    
        t = math.radians(theta)      
        a = math.cos(t) 
        b = math.sin(t)

        x0 = 0
        y0 = int((rho - x0 * a) / (b + 1e-6))

        x1 = w
        y1 = int((rho - x1 * a) / (b + 1e-6))
        cv2.line(out, (x0, y0), (x1, y1), (0, 0, 255), 2)   
    return out

def add_signature(img, sign_path = "Sign.bmp"):
    sign = cv2.imread(sign_path)    
    h, w = img.shape[:2]    
    sh, sw = sign.shape[:2] 
    x_start = w - sw        
    y_start = h - sh    
    for y in range(sh):
        for x in range(sw):
            b, g, r = sign[y, x]    
            if b > 240 and g > 240 and r > 240:
                continue
            img[y_start + y, x_start + x] = [b, g, r]   
    return img

img_list = ["fcu1.jpg", "fcu2.jpg", "Picture.jpg"]  
params = {"fcu1.jpg": 0.13, "fcu2.jpg": 0.1, "Picture.jpg": 0.15} 

for filename in img_list:
    img = cv2.imread(filename)  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    blur = gaussian(gray)                   
    gx, gy = my_sobel(blur)            
    mag, angle = mag_angle(gx, gy)  
    angle_q = quantize_angle(angle)         
    nms = NMS(mag, angle_q)                 
    strong, weak = double_threshold(nms)    
    canny_edge = hysteresis(strong, weak)   
    
    lines = hough_lines(canny_edge, acc_name = f"acc_{filename}", threshold = 200, sample_ratio = params[filename]) 
    edges_bgr = cv2.cvtColor(canny_edge, cv2.COLOR_GRAY2BGR)    
    result = draw_hough_lines(edges_bgr, lines)     
    result = add_signature(result, "Sign.bmp")      
    canny_bgr = cv2.cvtColor(canny_edge, cv2.COLOR_GRAY2BGR)    
    canny_bgr = add_signature(canny_bgr, "Sign.bmp")  
    cv2.imwrite(f"hough_{filename}", result)
    cv2.imwrite(f"canny_{filename}", canny_bgr)
