import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk

def load_images_from_folder(folder):
    
    images = []
    labels = []
    categories = ["stop", "right turn ahead", "left turn ahead", "speed limit"]
    for category in categories:
        category_path = os.path.join(folder, category)
        label = categories.index(category)
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64)) 
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)


train_images, train_labels = load_images_from_folder(r'C:\Users\aishwaryakh\Documents\Projects\Recognition of Highway Sign Boards\dataset_\training')
val_images, val_labels = load_images_from_folder(r'C:\Users\aishwaryakh\Documents\Projects\Recognition of Highway Sign Boards\dataset_\validation')
test_images, test_labels = load_images_from_folder(r'C:\Users\aishwaryakh\Documents\Projects\Recognition of Highway Sign Boards\dataset_\testing')

 
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0


train_labels = to_categorical(train_labels, 5)
val_labels = to_categorical(val_labels, 5)
test_labels = to_categorical(test_labels, 5)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

def predict_image(model, image_path,threshold=0.8):
    img = cv2.imread(image_path)        
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  
    predictions = model.predict(img)
    max_score=np.max(predictions)
    if max_score < threshold:
        return 'none'
    else:
        class_idx = np.argmax(predictions)
    categories = ["stop", "right turn ahead", "left turn ahead", "speed limit"]
    return categories[class_idx]

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        
        predicted_class = predict_image(model, file_path)
        result_label.config(text=f'Predicted class: {predicted_class}')


root = tk.Tk()
root.title("Traffic Sign Classifier")

panel = Label(root)
panel.pack(side="top", fill="both", expand="yes")

btn = tk.Button(root, text="Select an image", command=open_file)
btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

result_label = Label(root, text="Predicted class: None")
result_label.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

root.mainloop()
