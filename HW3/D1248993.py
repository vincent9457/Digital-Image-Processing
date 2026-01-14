import cv2                      
import numpy as np              

roi_selected = False           
roi_start = None               
roi_end = None                  
Rmin = Rmax = Gmin = Gmax = Bmin = Bmax = None  
Hmin = Hmax = Smin = Smax = Vmin = Vmax = None  
mode = 0                        
gmm_mu = None                   
gmm_var = None                  
gmm_w = None                    

def mouse_event(event, x, y, flags, param):
    global roi_start, roi_end, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)     
        roi_end = (x, y)        
    elif event == cv2.EVENT_MOUSEMOVE and roi_start is not None:
        roi_end = (x, y)        
    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)       
        roi_selected = True     

cap = cv2.VideoCapture(0)                       
cv2.namedWindow("Original")                     
cv2.setMouseCallback("Original", mouse_event)   

while True:
    ret, frame = cap.read()                     
    if not ret:
        break                                  
    display = frame.copy()                      
    if roi_start and not roi_selected:
        cv2.rectangle(display, roi_start, roi_end, (0, 255, 0), 2)

    if roi_selected and Rmin is None:
        x1, y1 = roi_start                     
        x2, y2 = roi_end                       
        x1, x2 = min(x1, x2), max(x1, x2)      
        y1, y2 = min(y1, y2), max(y1, y2)       
        roi = frame[y1:y2, x1:x2]               

        pixels = roi.reshape(-1, 3).astype(np.float32)  
        N = pixels.shape[0]                              
        mean = np.sum(pixels, axis=0) / N                
        diff = pixels - mean                             
        std = np.sqrt(np.sum(diff * diff, axis=0) / N)   
        k = 2
        Bmin, Bmax = int(mean[0] - k * std[0]), int(mean[0] + k * std[0])
        Gmin, Gmax = int(mean[1] - k * std[1]), int(mean[1] + k * std[1])
        Rmin, Rmax = int(mean[2] - k * std[2]), int(mean[2] + k * std[2])
        Bmin, Gmin, Rmin = max(0, Bmin), max(0, Gmin), max(0, Rmin)
        Bmax, Gmax, Rmax = min(255, Bmax), min(255, Gmax), min(255, Rmax)
        print("\nImproved RGB threshold:")      
        print(f"R: {Rmin} ~ {Rmax}")
        print(f"G: {Gmin} ~ {Gmax}")
        print(f"B: {Bmin} ~ {Bmax}")

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)         
        pixels_hsv = roi_hsv.reshape(-1, 3).astype(np.float32)  
        mean_hsv = np.sum(pixels_hsv, axis=0) / N               
        diff_hsv = pixels_hsv - mean_hsv                        
        std_hsv = np.sqrt(np.sum(diff_hsv * diff_hsv, axis=0) / N)  
        k = 2                       
        Hmin, Hmax = int(mean_hsv[0] - k * std_hsv[0]), int(mean_hsv[0] + k * std_hsv[0])
        Smin, Smax = int(mean_hsv[1] - k * std_hsv[1]), int(mean_hsv[1] + k * std_hsv[1])
        Vmin, Vmax = int(mean_hsv[2] - k * std_hsv[2]), int(mean_hsv[2] + k * std_hsv[2])
        Hmin, Smin, Vmin = max(0, Hmin), max(0, Smin), max(0, Vmin)
        Hmax, Smax, Vmax = min(179, Hmax), min(255, Smax), min(255, Vmax)
        print("HSV threshold:")                  
        print(f"H: {Hmin} ~ {Hmax}")
        print(f"S: {Smin} ~ {Smax}")
        print(f"V: {Vmin} ~ {Vmax}")

    if Rmin is not None and mode == 1:
        B = frame[:, :, 0]                      
        G = frame[:, :, 1]                      
        R = frame[:, :, 2]                   
        mask = ((R >= Rmin) & (R <= Rmax) & (G >= Gmin) & (G <= Gmax) & (B >= Bmin) & (B <= Bmax))
        result = np.zeros_like(frame)            
        result[mask] = frame[mask]               
        cv2.imshow("Result", result)             

    if Hmin is not None and mode == 2:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        mask = ((hsv[:, :, 0] >= Hmin) & (hsv[:, :, 0] <= Hmax) & (hsv[:, :, 1] >= Smin) & (hsv[:, :, 1] <= Smax) & (hsv[:, :, 2] >= Vmin) & (hsv[:, :, 2] <= Vmax))
        result = np.zeros_like(frame)           
        result[mask] = frame[mask]               
        cv2.imshow("Result", result)             

    if mode == 3 and Hmin is not None:
        K = 3                                    
        alpha = 0.01                             
        T = 0.7                                  
        k_sigma = 2.5                           
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        p = hsv                                  
        H, W, _ = p.shape                         
        hand_mask = ((hsv[:, :, 0] >= Hmin) & (hsv[:, :, 0] <= Hmax) & (hsv[:, :, 1] >= Smin) & (hsv[:, :, 1] <= Smax) & (hsv[:, :, 2] >= Vmin) & (hsv[:, :, 2] <= Vmax))
        
        if gmm_mu is None:
            gmm_mu = np.zeros((K, H, W, 3), np.float32)          
            gmm_var = np.ones((K, H, W, 3), np.float32) * 400    
            gmm_w = np.zeros((K, H, W), np.float32)              

            for k in range(K):
                gmm_mu[k] = p + np.random.randn(H, W, 3) * 5     
                gmm_w[k] = 1.0 / K                               

        diff = p[None, :, :, :] - gmm_mu                         
        dist = np.sum(diff * diff / (gmm_var + 1e-8), axis=3)    
        match = dist <= (k_sigma ** 2)                           
        matched_any = np.any(match, axis=0)                      

        for k in range(K):
            mk = match[k].astype(np.float32)                     
            gmm_w[k] = (1 - alpha) * gmm_w[k] + alpha * mk       
            rho = alpha * mk                                     
            gmm_mu[k] = (1 - rho[..., None]) * gmm_mu[k] + rho[..., None] * p   
            gmm_var[k] = (1 - rho[..., None]) * gmm_var[k] + rho[..., None] * (p - gmm_mu[k]) ** 2  

        no_match = ~matched_any                                  
        if np.any(no_match):
            min_k = np.argmin(gmm_w[:, no_match], axis=0)         
            ys, xs = np.where(no_match)                           
            for i in range(len(ys)):                              
                k = min_k[i]                                      
                y, x = ys[i], xs[i]                               
                gmm_mu[k, y, x] = p[y, x]                         
                gmm_var[k, y, x] = 400                            
                gmm_w[k, y, x] = 0.05                             

        gmm_w_sum = np.sum(gmm_w, axis=0, keepdims=True)  
        gmm_w /= (gmm_w_sum + 1e-8)    

        sigma_mean = np.sum(np.sqrt(gmm_var), axis=3) / 3         
        priority = gmm_w / (sigma_mean + 1e-8)                    
        order = np.argsort(-priority, axis=0)                     

        gmm_mu = np.take_along_axis(gmm_mu, order[..., None], axis=0)
        gmm_var = np.take_along_axis(gmm_var, order[..., None], axis=0)
        gmm_w = np.take_along_axis(gmm_w, order, axis=0)

        w_cum = np.cumsum(gmm_w, axis=0)                          
        bg_count = np.argmax(w_cum >= T, axis=0) + 1              

        background = np.zeros((H, W), bool)                       
        for k in range(K):
            in_bg = (k < bg_count)                                      
            diff = p - gmm_mu[k]                                        
            dist = np.sum(diff * diff / (gmm_var[k] + 1e-8), axis=2)    
            background |= (in_bg & (dist <= k_sigma ** 2))              

        foreground = ~background   
        #final_fg = foreground & hand_mask                                                         
        res = np.zeros_like(frame)                                 
        res[foreground] = frame[foreground]                         
        cv2.imshow("Result", res)                                   


    cv2.imshow("Original", display)                                 
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        mode = 1                                                    
        print("Mode 1: RGB segmentation")
    elif key == ord('2'):
        mode = 2                                                    
        print("Mode 2: HSV segmentation")
    elif key == ord('3'):
        mode = 3                                                    
        print("Mode 3: GMM segmentation")
    elif key == 27:
        break                                                      

cap.release()                                                        
cv2.destroyAllWindows()                                               
