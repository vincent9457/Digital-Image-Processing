# 數位影像處理作業
![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### HW0
1. 讀取一張照片，並輸入一個範圍 ( x , y , w , h )  
2. 將影像中從 ( x, y ) 開始，到 ( x + w , y + h ) 範圍的影像挖出來，存成一張新圖片
3. 兩種不同的實作相同功能：  
一、使用內建函數(crop or other function you want)  
二、只能使用 pixel 的讀取和寫入，禁止使用所有陣列相關操作和內建函數，只允許一個一個點取值/給值(配合迴圈)

### HW1
讀取彩色照片(不可使用 OpenCV 內建之縮放、旋轉等函數)，並包含以下功能:  
1. 將彩色圖轉成黑白
2. 輸入一個浮點數(0.1 ~ 2)，將影像放大或縮小(提供最近鄰內插和線性內插兩種選擇)
3. 輸入一個整數(0 ~ 360)，將影像(以中心)逆時針旋轉
4. 實作包含forward mapping和inverse mapping
5. 自動在輸出圖片的右下角加上簽名檔

### HW2
利用 OpenCV 寫出邊緣偵測: Canny Edge Detection 與線段偵測: Hough Transform ( 禁止使用 CannyEdge、HoughTransform 相關現成函數
 ) ，並在輸出的圖片右下角加上自己的簽名 ( 利用圖檔 )

Canny Edge Detection部分  
開始前先做 Gaussian Blur! (模糊化濾掉高頻/雜訊/細節)  
計算梯度用 Sobel、M 值計算用平方和開根號  
Non-Maximum Suppression 分成四個方向(上下、左右、正45度、負45度)，只保留同方向上連續點中的最大值。

Hough Transform部分  
隨機減少點的數量以加快計算速度。  
投票的時候附近 ϴρ 都投。(取樣問題/誤差問題)

### HW3
讀取 webcam 的影像，切割出手的部分。(切割、建立模型、比較這些部分禁止使用現成函數)
1. 利用 RGB / HSV color space 分割手部與其他部位
2. 實作 GMM。根據參考 frame 建立 GMM，每一個 frame 根據 GMM 切割前後景同時更新GMM。

**執行步驟**
1. 用滑鼠框選手部區域。
2. 按鍵盤數字來切換不同切割方式 (1.RGB 2.HSV 3.GMM )
3. 按 Esc 結束執行。
