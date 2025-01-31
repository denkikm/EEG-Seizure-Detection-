# EEG-Seizure-Detection-




# 🧠 EEG Seizure Detection using Machine Learning  
**Classifying EEG signals from different folders using Machine Learning models**  

---

## 📂 Project Structure  
```
📦 EEG_Seizure_Detection  
 ┣ 📂 data                 # Folder containing EEG data (Z, O, N, F, S)
 ┃ ┣ 📂 Z                  # EEG signals from class Z (e.g., healthy subjects)
 ┃ ┣ 📂 O                  # EEG signals from class O
 ┃ ┣ 📂 N                  # EEG signals from class N
 ┃ ┣ 📂 F                  # EEG signals from class F
 ┃ ┣ 📂 S                  # EEG signals from class S (e.g., seizure episodes)
 ┃ ┗ 📜 README.md          # Dataset information
 ┣ 📜 eeg_processing.py     # Python script for loading and preprocessing EEG data
 ┣ 📜 train_model.py        # Training script for Random Forest classification
 ┣ 📜 visualize_eeg.py      # Script for plotting EEG signals
 ┣ 📜 requirements.txt      # List of required Python packages
 ┣ 📜 README.md             # Project documentation
 ┗ 📜 .gitignore            # Files to be ignored in Git
```

---

## ⚡ Features  
✔ Automatically loads EEG `.txt` files from given folders  
✔ Preprocesses and standardizes EEG data  
✔ Extracts features and applies **Random Forest classifier**  
✔ Splits data into **train/test** sets  
✔ Plots EEG signals before and after preprocessing  

---

## 🔧 Installation  
### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/EEG_Seizure_Detection.git
cd EEG_Seizure_Detection
```

### 2️⃣ Install Required Packages  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run EEG Data Processing  
```bash
python eeg_processing.py
```

### 4️⃣ Train the Model  
```bash
python train_model.py
```

### 5️⃣ Visualize EEG Signals  
```bash
python visualize_eeg.py
```

---

## 📜 Dataset  
The EEG dataset consists of `.txt` files stored in 5 folders:  
- **Z:** Healthy subjects  
- **O:** Non-seizure EEG data  
- **N:** Interictal (between seizures)  
- **F:** Pre-seizure activity  
- **S:** Seizure episodes  

Each `.txt` file contains **4096 EEG signal samples**.  

---

## 📊 Model & Accuracy  
- **Classifier:** Random Forest (100 trees)  
- **Feature Scaling:** StandardScaler  
- **Data Split:** 80% Train, 20% Test  
- **Expected Accuracy:** ~97%  

---

## 📌 Code Overview  

### **📜 eeg_processing.py** (Load & Preprocess EEG Data)  
```python
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define data path
data_path = "data"

# EEG classes (folders)
folders = ["Z", "O", "N", "F", "S"]

def load_eeg_data():
    signals, labels = [], []
    for label, folder in enumerate(folders):
        folder_path = os.path.join(data_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.txt'):
                signal = np.loadtxt(os.path.join(folder_path, file))
                signals.append(signal)
                labels.append(label)
    return np.array(signals), np.array(labels)

# Load and preprocess data
X, Y = load_eeg_data()
X_scaled = StandardScaler().fit_transform(X)

print(f"Loaded {X.shape[0]} EEG signals from {len(folders)} classes")
```

---

### **📜 train_model.py** (Train Classifier)  
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(clf, "eeg_model.pkl")
```

---

### **📜 visualize_eeg.py** (Plot EEG Signals)  
```python
import matplotlib.pyplot as plt

def plot_eeg(signal, title="EEG Signal"):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()

# Plot first EEG sample
plot_eeg(X[0], "Raw EEG Signal")
plot_eeg(X_scaled[0], "Standardized EEG Signal")
```

---

## 🛠 Requirements  
📌 Install dependencies from `requirements.txt`:  
```txt
numpy
scipy
matplotlib
scikit-learn
joblib
```

---

## 🤝 Contributing  
🔹 Feel free to open issues or submit pull requests.  
🔹 Fork the repository and improve the project!  

---

## 📜 License  
🔓 This project is open-source under the **MIT License**.  

---

✅ **Now you're ready to run and test EEG classification!** 🚀  
🔹 **If you have any issues, feel free to ask.** 😊
