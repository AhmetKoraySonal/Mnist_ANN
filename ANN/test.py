import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
import glob
import pandas as pd
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
mnist_shape=(28,28)
model__path="model/final_minst_model.h5"
predict=[]
labels=[]
# Resim dosyasını yükle ve boyut kontrolü yap

def load_and_check_image(image_path):
    # Resmi aç
    img = Image.open(image_path).convert("L")
    
    # Resmin boyutunu kontrol et
    if img.size != (mnist_shape[0], mnist_shape[1]):  # 28x28 boyutlarında 1 kanal olmalı
        img = img.resize(mnist_shape[0], mnist_shape[1])
    
   
    # NumPy array'e çevir
    img_array = np.array(img)
    
    plt.imshow(img_array, cmap="gray")
    plt.axis("off")  # Eksenleri kaldır
    plt.show()

    
    img_array=img_array/255.0
    return  img_array.reshape(-1,mnist_shape[0]* mnist_shape[1])  # Modelin beklediği giriş formatına uygun hale getir



    
    
# Modeli yükle ve test et
def load_and_test_model(model_path, test_data, test_labels):
    global predict, labels
    # Modeli yükle
    model = load_model(model_path)
    
    # Test verisiyle tahminler yap
    predictions = model.predict(test_data)
    
    # Sınıf tahminlerini al
    predicted_classes = np.argmax(predictions, axis=1)
    
    test_labels = np.ravel(test_labels)
    predicted_classes = np.ravel(predicted_classes)
    
    predict=np.append(predict,predicted_classes)
    labels=np.append(labels,test_labels)
    
    # Test doğruluğunu hesapla
    accuracy = accuracy_score(test_labels, predicted_classes)
    
    # F1 skoru hesapla
    f1 = f1_score(test_labels, predicted_classes, average="micro")
    
    # Sonuçları yazdır
    print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    print(f"F1 Score: {f1 * 100:.2f}%\n")


# Komut satırı argümanlarını işle
def main():
    global predict, labels
    parser = argparse.ArgumentParser(description="Test a model on given data")
    parser.add_argument("--test_data", type=str, required=True, help="Test verisi dosyası (CSV) veya resimlerin bulunduğu dizin")
    parser.add_argument("--test_labels", type=str, help="Test etiketleri dosyasının yolu (Sadece resimler için)")

    args = parser.parse_args()

    # Model dosyasının yolu
    model_path = model__path
    
    test_images = []
    
    # Eğer test verisi CSV dosyasıysa
    if os.path.isfile(args.test_data) and args.test_data.endswith(".csv"):
        df = pd.read_csv(args.test_data, header=None,low_memory=False)  # Başlıksız CSV dosyası
        test_labels = df.iloc[1:, 0].values  # İlk sütunu etiket olarak al.numpya çevrilir
        test_images = df.iloc[1:, 1:].values  # Kalan sütunları giriş verisi olarak al

        # Verileri uygun şekilde şekillendir
        test_images = test_images.astype(np.float32)/255.0  # float32'ye dönüştürme
        
        test_images = test_images.reshape(-1, mnist_shape[0]*mnist_shape[1])  # 28x28x1 yerine 784 uzunluğunda vektör
        print(f"Test images shape: {test_images.shape}")  
        print(f"Test labels shape: {test_labels.shape}") 
        
     
    # Eğer test verisi bir klasörse
    elif os.path.isdir(args.test_data):
        if not args.test_labels:
            raise ValueError("If test data is images, you must provide --test_labels")

        if not os.path.exists(args.test_labels):
            raise FileNotFoundError(f"Test labels file not found at {args.test_labels}")

        with open(args.test_labels, "r", encoding="utf-8") as f:
            test_labels = [line.strip() for line in f]
        
        
        image_paths = glob.glob(os.path.join(args.test_data, "*.jpg")) + glob.glob(os.path.join(args.test_data, "*.png"))
        
        for image_path in image_paths:
            test_img = load_and_check_image(image_path)
            test_images.append(test_img)
        
        test_images = np.vstack(test_images)

    else:
        raise ValueError("Invalid test data path. Provide either a CSV file or a directory containing images.")

    # Test etiketlerini one-hot formatına dönüştür
    test_labels = np.array(test_labels, dtype=int)
    # Modeli yükleyip testi gerçekleştir
    load_and_test_model(model_path, test_images, test_labels)
    
    show_prediction = input("Do you want the predictions (E/H): ").strip().lower()
    predict_len=len(predict)
    if show_prediction == "e":
        number_of_predictions=int(input(f"Kaç tanesini gormek istersiniz? Max:({predict_len})\n"))
        if number_of_predictions>predict_len:
            raise ValueError("There aren't that many predictions available")
        
        selected_indices = np.random.choice(len(predict), number_of_predictions, replace=False)
        selected_predictions = predict[selected_indices]
        selected_labels=labels[selected_indices]
        for i in range(len(selected_predictions)):
            print(f"Label : {selected_labels[i]} Tahmin: {selected_predictions[i]}")

    
if __name__ == "__main__":
    main()
