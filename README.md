# Handwriting-Recognition
Handwritten Text Recognition using CRNN with Profile Normalization and RWGD Augmentation. A complete end-to-end OCR pipeline built in PyTorch for converting handwritten images into digital text, with CER/WER evaluation and memory-efficient training on local systems.

 
# 📝 Handwritten Text Recognition using CRNN + PN + RWGD

This project focuses on converting **handwritten text images into editable digital text** using a deep learning model based on **CRNN (CNN + RNN + CTC Loss)**.  
It also uses **Profile Normalization (PN)** and **RWGD augmentation** to improve accuracy on real-world handwritten data.

This project is designed to run on a **local system** in a **memory-efficient way**.


## 🎯 Project Objective

To build an **end-to-end handwritten text recognition system** that:
- Takes a handwritten text image as input
- Preprocesses it using PN and RWGD
- Predicts the corresponding digital text using a CRNN model

---

## 🧠 Model Architecture

Input Image → Profile Normalization → RWGD Augmentation
→ CNN → BiLSTM → CTC Loss → Predicted Text


- **CNN** → extracts visual features  
- **BiLSTM** → learns sequence patterns  
- **CTC Loss** → aligns predictions without character-level mapping  

## ✨ Key Features

✅ Profile Normalization (PN)  
✅ RWGD augmentation  
✅ CRNN deep learning model  
✅ Character & Word Error Rate evaluation (CER & WER)  
✅ Memory-efficient training  
✅ Fast-mode for quick testing  
✅ GPU & CPU supported  

## 🛠 Tech Stack

| Category | Tools |
|----------|--------|
| Language | Python |
| Deep Learning | PyTorch |
| Image Processing | OpenCV, NumPy |
| Visualization | Matplotlib |
| Data Handling | Glob, Regex, Pandas |



## 📁 Dataset Structure

dataset/
├── Images/
│   ├── train2011-xxx.jpg
│   ├── eval2011-xxx.jpg
│
├── Transcriptions/
│   ├── train2011-xxx.txt
│   ├── eval2011-xxx.txt


Each image must have a corresponding transcription.

## ⚙️ Installation

bash
pip install torch torchvision torchaudio opencv-python numpy matplotlib pandas tqdm

## 🚀 How to Run the Project

### 1️⃣ Load Transcriptions

Run the transcription loader cell (Cell 4) to build:

* `TRANS_MAP`
* Character vocabulary (`stoi`, `itos`)


### 2️⃣ Train the Model

Run Cell 12:

python
model, metrics, voc = run_experiment(
    augment_type='rwgd',
    epochs=7,
    batch_size=8,
    use_pn=True,
    channel_mode='gray',
    fast_mode=False
)


💡 Use `fast_mode=True` for quick debugging.


### 3️⃣ Visualize Predictions

Run Cell 13 to see:

* Original Image
* After Profile Normalization
* After RWGD
* Predicted text with CER & WER


## 📊 Evaluation Metrics

| Metric | Meaning              |
| ------ | -------------------- |
| CER    | Character Error Rate |
| WER    | Word Error Rate      |

* **Best value = 0.0**
* **Worst value = 1.0**


## 🧩 Important Concepts

### ✅ Profile Normalization (PN)

Straightens the text baseline and normalizes writing height.

### ✅ RWGD Augmentation

Simulates handwriting variation using smooth random warping.

### ✅ CTC Loss

Allows sequence prediction without exact alignment between image and text.


## 💾 Model Saving

Model checkpoints are saved automatically:

crnn_rwgd_epoch1.pth
crnn_rwgd_epoch2.pth
 

To load:

```python
model.load_state_dict(torch.load("crnn_rwgd_epoch7.pth"))
```

---

## 🧪 Sample Output

| Image             | Prediction                       |
| ----------------- | -------------------------------- |
| Handwritten Image | "courrier vos disponibilités..." |


## ⚠️ Common Issues & Fixes

| Issue         | Reason             | Fix                     |
| ------------- | ------------------ | ----------------------- |
| CER/WER = 1.0 | Model not learning | Check labels & training |
| Kernel crash  | High memory usage  | Reduce batch size       |
| PN error      | PN not defined     | Run PN cell first       |
| RWGD error    | RWGD not defined   | Define RWGD_simple      |
| CUDA OOM      | GPU memory full    | Use batch_size = 2      |

---

## 🧑‍🎓 Note

This project was built as a **learning project** to understand:

* OCR systems
* Deep learning for sequences
* Data preprocessing for vision tasks

Everything is implemented to work on a **normal laptop**, without high-end GPUs.


## 🔮 Future Scope

* Transformer-based HTR
* Real-time camera recognition
* Spell-correction postprocessing
* Streamlit web app


## 👩‍💻 Author

**Manikarnika Yadav**
M.Tech (Cyber Physical Systems) – IIT Jodhpur
AI & Computer Vision Enthusiast


## 📜 License

This project is for **educational and research use only**.
Free to use with proper credit.
