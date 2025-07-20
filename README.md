Sure! Here's a professional and complete `README.md` file for your **Face Mask Detection System** project, including details about:

* What the project does
* How it works
* Requirements
* Training and live detection
* Sample code
* Licensing info

---

### ✅ `README.md`

````markdown
# 😷 Face Mask Detection Using CNN + OpenCV + Keras + TTS

This project is a real-time **Face Mask Detection System** that uses a convolutional neural network (CNN) to detect whether a person is wearing a face mask or not via webcam. It also includes **text-to-speech** announcements using `pyttsx3`.

---

## 📌 Features

- Classifies people as **with mask** or **without mask** in real-time
- Trained on a custom dataset using CNN from scratch
- Uses **OpenCV** for face detection and video feed
- **Text-to-Speech** alerts for mask status using `pyttsx3`
- Timestamp overlay for every frame
- Live detection with `haarcascade_frontalface_default.xml`

---

## 🧰 Tech Stack

- Python
- Keras / TensorFlow
- OpenCV
- Pyttsx3 (Text to Speech)
- NumPy
- H5py

---

## 🛠️ Installation

Install required Python packages:

```bash
pip install opencv-python keras tensorflow pyttsx3 numpy h5py
````

Also, ensure you have the Haar Cascade XML file for face detection:

```text
haarcascade_frontalface_default.xml
```

---

## 🧠 Model Architecture

* 3 Convolutional Layers with ReLU activation
* MaxPooling after each Conv2D layer
* Flatten → Dense(100) → Dense(1) with sigmoid
* Binary classification (mask = 0, no mask = 1)

---

## 🏋️ Training the Model

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D())
...

# Data Generators
train_datagen = ImageDataGenerator(rescale=1./255, ...)
train_generator = train_datagen.flow_from_directory("Train/")

# Train
model.fit(train_generator, validation_data=val_generator, epochs=10)
model.save('mymodel.h5')
```

---

## 📹 Real-Time Mask Detection

```python
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mymodel = load_model('mymodel.h5')
...

# Predict mask vs no mask
if pred == 1:
    engine.say("Person without mask detected")
else:
    engine.say("Person with mask detected")
```

* Press `Q` to quit webcam feed

---

## 📁 Directory Structure

```text
FaceMaskDetector/
├── Train/
├── Test/
├── Validation/
├── haarcascade_frontalface_default.xml
├── mymodel.h5
├── face_mask_detector.py
```

---

## 📢 Voice Feedback

Implemented using `pyttsx3`. Alerts:

* "Person with mask detected"
* "Person without mask detected"

Cooldown of **3 seconds** added between announcements to prevent repetition.

---

## ✅ How to Run

1. Train the model or use `mymodel.h5`
2. Run the Python script:

```bash
python face_mask_detector.py
```

3. Allow webcam access
4. Look at the camera — feedback will be shown and spoken

---

## 🖼 Sample Output

* Bounding boxes around detected faces
* Green = Mask, Red = No Mask
* Timestamp shown
* Audio announcement

---

## 📄 License

This project is for educational and research use only.

---

## 👨‍💻 Author

**Karan**
Modifications and voice alert by **Subbarayudu Chittiboina**

---

## 🙋 Need Help?

If you're facing issues with:

* Webcam not opening
* No prediction
* Missing files

Feel free to raise an issue or contact me.

```

---

