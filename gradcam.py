import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 先定義 Grad-CAM 函數
def grad_cam(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = np.mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap 

def overlay_heatmap(heatmap, img_array, alpha=0.4):
    img = np.squeeze(img_array)
    img = np.uint8(255 * img)  # 將圖像轉換為 uint8 類型
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # 確保 heatmap 是 uint8 類型
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return superimposed_img
 
def preprocess_image(image_path, img_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def show_gradcam(image_path, model, last_conv_layer_name):
    img_array = preprocess_image(image_path)
    heatmap = grad_cam(model, img_array, last_conv_layer_name)
    superimposed_img = overlay_heatmap(heatmap, img_array)
    
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show() 

# 取得測試資料夾中的所有圖片
test_dir = "C:/Users/ycliou/Desktop/xray/8020/test"
categories = ["n", "p"]

# 顯示所有圖片的 Grad-CAM
def show_gradcam_for_all_images_in_folder(test_dir, model, last_conv_layer_name, output_folder=None):
    for category in categories:
        path = os.path.join(test_dir, category)
        for img in os.listdir(path):
            if img.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(path, img)
                print(f"正在處理: {image_path}")
                # 顯示 Grad-CAM 視覺化圖像
                show_gradcam(image_path, model, last_conv_layer_name)

                # 如果指定了輸出資料夾，則保存 Grad-CAM 圖像
                if output_folder:
                    save_gradcam_image(image_path, model, last_conv_layer_name, output_folder)

# 儲存 Grad-CAM 圖像
def save_gradcam_image(image_path, model, last_conv_layer_name, output_folder):
    # 獲取圖像的基本檔名
    image_name = os.path.basename(image_path)
    
    # 獲取 Grad-CAM 視覺化圖像
    img_array = preprocess_image(image_path)
    heatmap = grad_cam(model, img_array, last_conv_layer_name)
    superimposed_img = overlay_heatmap(heatmap, img_array)
    
    # 檢查輸出資料夾是否存在，若不存在則創建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 儲存結果圖像
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, superimposed_img)  # 使用 OpenCV 儲存圖像

    print(f"已儲存圖像: {output_path}")

# 假設模型已經訓練完成，並指定最後一層卷積層的名稱
last_conv_layer_name = "last_conv_layer"

# 指定結果保存的資料夾
output_folder = "C:/Users/ycliou/Desktop/xray/gradcam_results"

# 顯示所有測試集中的圖像，並儲存結果
show_gradcam_for_all_images_in_folder(test_dir, model, last_conv_layer_name, output_folder)