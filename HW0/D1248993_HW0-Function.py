import cv2
import time

img = cv2.imread('Picture.jpg', -1)     
x, y, w, h = map(int, input().split())  
start = time.time()
roi = img[y:y+h, x:x+w]                 
cv2.imwrite("NewPicture.jpg", roi)      
end = time.time()
print("內建函數耗時:", end - start, "秒")
