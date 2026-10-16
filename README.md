<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:134E5E,100:71B280&height=220&section=header&text=Human%20Activity%20Recognition&fontSize=44&fontColor=ffffff&animation=fadeIn&fontAlignY=36&desc=Early%20Fusion%20of%20Multimodal%20Wearable%20Sensors%20-%20CogAge%20Dataset&descSize=18&descAlignY=56" width="100%" alt="Human Activity Recognition banner"/>

<a href="#-key-results">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1200&color=71B280&center=true&vCenter=true&width=720&lines=Smartphone+%2B+Smartwatch+%2B+Smart+Glasses;9+sensor+streams+fused+into+27+channels;LSTM+vs+Simple+RNN+vs+CNN-LSTM;Subject-dependent+vs+subject-independent;Best+test+accuracy%3A+81.82%25+(LSTM)" alt="Typing animation"/>
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

![Task](https://img.shields.io/badge/Task-12--class%20HAR-134E5E?style=flat-square)
![Fusion](https://img.shields.io/badge/Fusion-Early%20(feature--level)-71B280?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-CogAge-2c5364?style=flat-square)
![Course](https://img.shields.io/badge/M.Phil%20AI-PUCIT-1f6f8b?style=flat-square)

</div>

---

## 📌 Overview

People carry several sensors at once: a phone in the pocket, a watch on the wrist, and sometimes smart glasses. This project asks a simple question: **if we fuse all of those signals at the input level, how well can a sequence model recognise what a person is doing?**

Nine sensor streams from three devices are resampled to the same length and concatenated into a single **100 × 27** time series per sample (*early fusion*). Three recurrent architectures are trained on it and evaluated under two protocols: one where the same people appear in training and testing, and a harder one where the model is tested on **a person it has never seen**.

<table>
<tr>
<td align="center"><b>3</b><br/>devices</td>
<td align="center"><b>9</b><br/>sensor streams</td>
<td align="center"><b>27</b><br/>fused channels</td>
<td align="center"><b>12</b><br/>activities</td>
<td align="center"><b>436</b><br/>samples</td>
<td align="center"><b>3</b><br/>models compared</td>
</tr>
</table>

---

## 🏆 Key Results

| Model | Subject-Dependent (test) | Subject-Independent (test on unseen person) |
|:--|:--|:--|
| **LSTM** (2 layers) | `████████░░` **81.82%** 🥇 | `█░░░░░░░░░` 13.55% |
| **Simple RNN** (2 layers) | `████████░░` 78.79% | `██░░░░░░░░` **22.90%** |
| **CNN + LSTM** | `███████░░░` 69.70% | `██░░░░░░░░` 22.43% |

> [!IMPORTANT]
> **The generalization gap is the real finding.** All three models reach 70–82% when both subjects are mixed in training and testing, but drop to 13–23% when trained on one person and tested on another (random guessing over 12 classes would be ≈ 8.3%). The networks learn *how a specific person moves*, not just *what the activity is*. Closing this gap with more subjects, per-user normalisation or domain adaptation is the natural next research step.

---

## 🔄 Pipeline

```mermaid
flowchart LR
    A["📱 Smartphone<br/>accelerometer · gyroscope<br/>gravity · linear acc · magnetometer"] --> D
    B["⌚ Smartwatch<br/>accelerometer · gyroscope"] --> D
    C["👓 JINS smart glasses<br/>accelerometer · gyroscope"] --> D
    D["Linear resampling<br/>each stream → 100 steps × 3 axes"] --> E["Early fusion<br/>concatenate → 100 × 27"]
    E --> F{"Sequence models"}
    F --> G["LSTM"]
    F --> H["Simple RNN"]
    F --> I["CNN + LSTM"]
    G --> J["12-class activity prediction"]
    H --> J
    I --> J
```

---

## 📂 Dataset

**CogAge** multimodal wearable dataset, 12-activity subset. Two subjects were selected for this study, giving **436 fused samples**.

<details>
<summary><b>🎯 The 12 activities</b></summary>
<br/>

| | | | |
|:--|:--|:--|:--|
| 🙇 Bending | 📦 Bring | 🧹 Clean Floor | 🧽 Clean Surface |
| 🥤 Drink | 🙌 Dry Off Hands | 🍬 Eat Small Thing | 🛌 Lying |
| 🪑 Sitting | 🏋️ Squatting | 🧍 Standing | 🚶 Walking |

</details>

<details>
<summary><b>📡 Sensor streams (9 × 3 axes = 27 channels)</b></summary>
<br/>

| Device | Sensors | Channels |
|:--|:--|:--:|
| Smartphone | Accelerometer, Gyroscope, Gravity, Linear Accelerometer, Magnetometer | 15 |
| Smartwatch | Accelerometer, Gyroscope | 6 |
| JINS smart glasses | Accelerometer, Gyroscope | 6 |
| **Total** | **9 streams** | **27** |

Raw streams arrive at different rates (roughly 100 to 1,000 readings per event), so each stream is linearly interpolated to a fixed **100 timesteps** before fusion.

</details>

---

## 🧪 Experimental Protocols

| Protocol | Train | Validation | Test | What it measures |
|:--|:--|:--|:--|:--|
| **Subject-dependent** | 305 samples | 65 | 66 | Stratified 70 / 15 / 15 split over both subjects |
| **Subject-independent** | 177 (subject A) | 45 (subject A) | 214 (subject B) | Generalization to a completely unseen person |

---

## 🧠 Models

<details open>
<summary><b>Architectures</b></summary>
<br/>

| Model | Architecture |
|:--|:--|
| **LSTM** | Input (100 × 27) → 2 × LSTM (64 units, dropout 0.3) → last hidden state → Linear (12) |
| **Simple RNN** | Input (100 × 27) → 2 × RNN (64 units, dropout 0.3) → last hidden state → Linear (12) |
| **CNN + LSTM** | Conv1D (27 → 64, k = 3) → ReLU → MaxPool (2) → LSTM (64) → Dropout (0.3) → Linear (12) |

</details>

<details>
<summary><b>⚙️ Training configuration</b></summary>
<br/>

| Setting | Value |
|:--|:--|
| Loss | Cross-entropy |
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 32 |
| Epochs | 30 |
| Model selection | Best validation accuracy checkpoint |
| Random seed (splits) | 42 |
| Environment | Google Colab |

</details>

---

## 🚀 Getting Started

```bash
git clone https://github.com/RamzanPUCIT/pattern-recognition-early-fusion-har.git
cd pattern-recognition-early-fusion-har
pip install torch numpy pandas scikit-learn matplotlib
```

1. Open `earlyfusion.ipynb` in **Google Colab**.
2. Place the CogAge 12-activity data in Google Drive at `MyDrive/12Activities_Dataset/` (the dataset is not included in this repository).
3. Run the cells in order: data loading → resampling and fusion → subject-dependent experiments → subject-independent experiments.

---

## 🗂️ Repository Structure

```
pattern-recognition-early-fusion-har/
├── earlyfusion.ipynb   # Full pipeline: loading, fusion, training, evaluation
├── Report.pdf          # Detailed write-up of the method and results
└── README.md
```

---

## 🔭 Future Work

- [ ] Leave-one-subject-out evaluation across all 8 CogAge subjects
- [ ] Per-subject normalisation and domain adaptation to close the generalization gap
- [ ] Compare early fusion with late (decision-level) fusion
- [ ] Transformer / temporal-convolution baselines
- [ ] Extend the same evaluation protocol to contactless sensing (e.g. WiFi CSI)

---

## 👨‍💻 Author

**Muhammad Ramzan** · M.Phil Artificial Intelligence, PUCIT (University of the Punjab), Lahore

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ramzan--ai-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ramzan-ai)
[![GitHub](https://img.shields.io/badge/GitHub-RamzanPUCIT-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RamzanPUCIT)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:founders786@gmail.com)

<div align="center">

⭐ **If this project helped you, consider giving it a star!** ⭐

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:134E5E,100:71B280&height=120&section=footer" width="100%" alt="footer"/>

</div>
