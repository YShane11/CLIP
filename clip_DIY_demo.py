import torch 
import clip
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import cv2
import numpy as np

def extract_foreground(image_path, rect=None, iterations=10, save_path=None): # 負責將圖片分出前景與背景
    image = cv2.imread(image_path) # 讀取圖片
    
    if image is None:
        raise ValueError("找不到照片") # 防錯誤
    if rect is None:
        h, w = image.shape[:2] # 取得圖片的高和寬
        rect = (10, 10, w - 20, h - 20) # 會需要設定這個rect 是因為grabCut會需要將物體限縮在一個範圍內 但由於沒有確定物體 這邊一律設在邊邊
        
    mask = np.zeros(image.shape[:2], np.uint8) # 設定一個跟圖片一樣大小的矩陣 要來儲存遮罩
    bgd_model = np.zeros((1, 65), np.float64) # 用來儲存背景模型資訊
    fgd_model = np.zeros((1, 65), np.float64) # 用來儲存前景模型資訊
    
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT) # 執行 GrabCut
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8") # 將確定前景和可能前景標記為 1，其餘為 0
       
    if save_path:
        mask_output = (mask2 * 255).astype("uint8")  # 將遮罩擴展至 0-255 範圍以顯示為圖像(會是黑白的)
        cv2.imwrite(save_path, mask_output) # 儲存起來
        print(f"前景提取的圖片以儲存於: {save_path}")
    
    return mask2 # 回傳遮罩的部分

device = "cuda" if torch.cuda.is_available() else "cpu" # 設置設備 這便要確定有沒有安裝好cuda
model, preprocess = clip.load("ViT-L/14@336px", device=device)
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'] 為clip的可用模型

image_path = "data/test_3.jpg" # 這邊是需要處理的圖片之路徑
image = Image.open(image_path)
width, height = image.size 

labels = ["background","dog","cat","horse"] # 設定圖片中可能包含的類別 在這個程式中background必須放在第一項
background_threshold = 0.4  # 背景的閾值

# 設定每個標籤遮罩的顏色
label_colors = {
    f'{labels[1]}': (1, 0, 0, 0.3),  # 紅色
    f'{labels[2]}': (0, 0, 1, 0.3),   # 藍色
    f'{labels[3]}': (0, 1, 0, 0.3),   # 綠色
}

patch_sizes = [64,128,256] # 設定要切的三個尺度 這邊64、128、256為論文實驗過後較為推薦的
weights = [0.6,0.3,0.1]  # 為每個尺度分配權重，總和要確定是1

fusion_map = np.zeros((len(labels), height, width)) # 這邊是要用來儲存最後各尺度特徵權重

for patch_size_index,patch_size in enumerate(patch_sizes): 
    start_time = time.time() # 要用來計每個尺度要切多久
    patch_size_x = patch_size
    patch_size_y = patch_size

    num_patches_y = 0 # x切幾次patch(後面要用到)
    num_patches_x = 0 # y切幾次patch(後面要用到)
    
    image_patches = [] # 用來儲存預處理過後的小區塊
    for y in range(0, height, patch_size_y): 
        num_patches_y += 1
        for x in range(0, width, patch_size_x):
            if y == 0:
                num_patches_x += 1
            patch = image.crop((x, y, x + patch_size_x, y + patch_size_y)) # 根據patch_size切割成每個小區塊
            image_patches.append(preprocess(patch).unsqueeze(0).to(device)) # 將切割完的小區塊預處理後儲存起來
    
    
    text_inputs = clip.tokenize(labels).to(device) # 將標籤進行token化，也就是embedding
    segmentation_results = [] # 用來儲存每個小區塊的分割結果(哪個標籤)
    softmax_results = [] # 儲存每個小區塊的 softmax 機率

    for image_input in image_patches:
        with torch.no_grad(): # 禁用梯度計算，為了提高推理速度
            image_features = model.encode_image(image_input) # 將小區塊編碼成特徵向量
            text_features = model.encode_text(text_inputs) # 將標籤文本編碼成特徵向量
            logits_per_image, _ = model(image_input, text_inputs) # 將小區塊跟文本塞到模型裡 logits_per_image是每個區塊的分數
            probs = logits_per_image.softmax(dim=-1).cpu().numpy() # 計算softmax機率(各標籤機率)，然後轉成NumPy

        best_label, best_prob = max(zip(labels, probs[0]), key=lambda x: x[1]) # 找出最可能的標籤及其機率
        
        if best_prob < background_threshold:  # 設定背景閥值
            best_label = "background" # 如果那個區塊的最高分數沒有高過閥值 就會被當成閥值
            probs[0] = [probs[0][0]] + [0] * (len(probs[0]) - 1) # 其他標籤分數會被歸零，只保留背景的分數
        segmentation_results.append(best_label) # 將最佳標籤結果添加到分割結果中
        softmax_results.append(probs[0]) # 將所有標籤的 softmax 機率添加到 softmax_results 中(後續操作會用到)

# ======================================fusion_map 處理===========================================
    
    softmax_results = np.array(softmax_results).transpose().tolist() # 為了將這個array整理成「標籤數x高x寬」的矩陣，先轉置陣列再轉成列表

    for i in range(len(labels)):
        reshaped_softmax_results = np.array(softmax_results[i]).reshape(num_patches_y, num_patches_x) 
        # 將標籤的 softmax 分數重塑為 「切割數量的高」乘「切割數量的寬」
        repeated_result = np.repeat(np.repeat(reshaped_softmax_results, repeats=patch_size, axis=0), repeats=patch_size, axis=1)
        # 將 reshaped_softmax_results 變回原圖的長乘寬(也就是說每個pixel都可以記錄分數)
        repeated_result = repeated_result[:height, :width] # 由於原圖可能不一定是尺度的倍數 所以要再對齊一下
        fusion_map[i] += weights[patch_size_index] * repeated_result # 乘上每個patch_size分別的權重後，加到最後總整權重的fusion_map矩陣裡

# ================================================================================================
   
    segmentation_map = np.array(segmentation_results).reshape(num_patches_y, num_patches_x)
    # 將小區塊的分割結果重塑為「切割數量的高」乘「切割數量的寬」(這邊是紀錄每個區塊分數最好的標籤)
    
    fig, ax = plt.subplots(figsize=(8, 8)) 
    ax.imshow(image) # 顯示原始圖片作為背景

    for y in range(num_patches_y): 
        for x in range(num_patches_x):
            label = segmentation_map[y, x] # 取得當前小區塊的標籤
            if label != "background":  # 只標記非背景區域                 
                rect = patches.Rectangle(
                    (x * patch_size_x, y * patch_size_y), patch_size_x, patch_size_y,
                    linewidth=1, edgecolor='none', facecolor=label_colors[label]
                ) # 為非背景的小區塊使用相對應的顏色添加半透明的遮罩
                ax.add_patch(rect) # 添加到原始圖片上

    plt.axis("off") # 要把座標軸隱藏掉 不然會很奇怪
    output_path = f"output/{patch_size}x{patch_size}.png"  # 指定輸出檔案路徑和名稱
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 確保目錄存在
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)  # 儲存
    plt.close(fig)  # 關閉圖表，釋放記憶體
    
    end_time = time.time()  # 結束計時
    print(f"執行 {patch_size}x{patch_size} 區塊耗時: {end_time - start_time:.2f} 秒")  # 顯示當前尺度的處理時間


# ======================================fusion_map 最後處理===========================================

# fusion_map 是 (標籤數 x 高 x 寬)
max_indices = np.argmax(fusion_map, axis=0) # 將 fusion_map 中最高分數的標籤分數取出，變成 (最高的標籤分數的index x 高 x 寬)

result_labels = np.empty(max_indices.shape, dtype=object) # 創一個空的陣列，用來儲存結果標籤(最高的分數對應的類別)

for i in range(max_indices.shape[0]):
    for j in range(max_indices.shape[1]):
        result_labels[i, j] = labels[max_indices[i, j]] # 將對應的類別名稱儲存進去

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image) # 顯示原始背景圖片

for i in range(result_labels.shape[0]):
    for j in range(result_labels.shape[1]):
        label = result_labels[i, j] # 取得當前pixel的標籤
        if label in label_colors:
            rect = patches.Rectangle(
                (j, i), 1, 1,
                linewidth=0, edgecolor='none', facecolor=label_colors[label]
            ) # 使用標籤對應顏色繪製矩形
            ax.add_patch(rect) # 添加到原始圖片上

plt.axis("off") # 要把座標軸隱藏掉 不然會很奇怪
output_path = "output/final_result.png" # 指定輸出檔案路徑和名稱
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 確保目錄存在
plt.savefig(output_path, bbox_inches="tight", pad_inches=0)  # 儲存
plt.close(fig)  # 關閉圖表，釋放記憶體

# ======================================最後結合上extract_foreground去塞選===========================================

foreground_mask = extract_foreground(image_path,save_path="output/foreground_test.png")
# 使用 extract_foreground 提取前景背景的遮罩矩陣
fusion_map = fusion_map * foreground_mask # 將前景遮罩跟fusion_map相乘(非前景部分會被砍掉)
 
 
max_indices = np.argmax(fusion_map, axis=0) # 將 fusion_map 中最高分數的標籤分數取出，變成 (最高的標籤分數的index x 高 x 寬)
 
result_labels = np.empty(max_indices.shape, dtype=object) # 創一個空的陣列，用來儲存結果標籤(最高的分數對應的類別)

for i in range(max_indices.shape[0]):
    for j in range(max_indices.shape[1]):
        result_labels[i, j] = labels[max_indices[i, j]] # 將對應的類別名稱儲存進去


fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image) # 顯示原始背景圖片

for i in range(result_labels.shape[0]):
    for j in range(result_labels.shape[1]):
        label = result_labels[i, j] # 取得當前pixel的標籤
        if label in label_colors:
            rect = patches.Rectangle(
                (j, i), 1, 1,
                linewidth=0, edgecolor='none', facecolor=label_colors[label]
            ) # 使用標籤對應顏色繪製矩形
            ax.add_patch(rect) # 添加到原始圖片上

plt.axis("off") # 要把座標軸隱藏掉 不然會很奇怪
output_path = "output/final_result_with_mask.png" # 指定輸出檔案路徑和名稱
os.makedirs(os.path.dirname(output_path), exist_ok=True) # 確保目錄存在
plt.savefig(output_path, bbox_inches="tight", pad_inches=0)   # 儲存
plt.show()