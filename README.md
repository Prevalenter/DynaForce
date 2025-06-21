# **DynaForce: Sensorless Contact Estimation for Dexterous Manipulation**

🔗 [Project Homepage](https://sites.google.com/view/dex-sensorless/)

**DynaForce** is an open-source framework that estimates contact forces from joint torques using robotic dynamics, enabling **sensorless force estimation** in dexterous manipulation tasks.

> **Note:** This paper is currently under review. Additional code and resources will be released upon acceptance.

<div style="text-align:center">
    <img src="https://github.com/user-attachments/assets/195537c4-59bb-42a1-9ac5-b6a1b582fd4a" width="100%">
</div>

---

## 🚧 TODO

- [ ] Identify base inertia parameters for the robot hand  
- [x] Implement momentum-based collision detection  
- [x] Expand PyQt UI for easier finger/link selection  
- [x] Validate real-world force estimation via collision experiments  
- [ ] Publish training code for imitation learning

---

## ⚙️ Installation

- Python 3.8 is required.
- If you want to try the **interactive simulation demo**, make sure to install **PyQt5**.
- All package versions are listed in [`utils/python_version.txt`](utils/python_version.txt).

---

## 🚀 Getting Started

### 🔍 Run Identification
*(Add detailed command or description here if available)*

---

### 🧪 Try in Simulation with UI (**Recommended**)

```bash
cd ident/sim/
python sim_interaction_qt.py
```

---

### 📡 Momentum-Based Force Estimation (Real World)

```bash
cd ident/real/fore_calibrate/
python pred_momentum.py --K 2.5
```

---
