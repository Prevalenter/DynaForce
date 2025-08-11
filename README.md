# **Dexsensorless: Sensorless Contact Estimation for Dexterous Manipulation**

🔗 [Project Homepage](https://sites.google.com/view/dex-sensorless/)

**Dexsensorless** is an open-source framework that estimates contact forces from joint torques using robotic dynamics, enabling **sensorless force estimation** in dexterous manipulation tasks.

> **Note:** This paper is currently under review. Additional code and resources will be released upon acceptance.

<div style="text-align:center">
    <img src="https://github.com/user-attachments/assets/28d6ae04-2db4-4f01-ad83-5c735a70d5cc" width="100%">
</div>

---

## 🚧 TODO

- [x] Identify base inertia parameters for the robot hand  
- [x] Implement momentum-based collision detection  
- [x] Expand PyQt UI for easier finger/link selection  
- [x] Validate real-world force estimation via collision experiments  
- [ ]  Training code for imitation learning

---

## ⚙️ Installation

First, install the [SymPyBotics](https://github.com/cdsousa/SymPyBotics)
```bash
git clone https://github.com/cdsousa/SymPyBotics.git
cd sympybotics
python setup.py install
```

Then, clone the repository and navigate into the directory:
```bash
git clone https://github.com/Prevalenter/DynaForce.git
```

- Python 3.8 is required.
- If you want to try the **interactive simulation demo**, make sure to install **PyQt5**.
- All package versions are listed in [`utils/python_version.txt`](utils/python_version.txt).

If you have any questions, please submit an issue.

## 📖 Project Overview

The repository is organized to support both simulation and real-world experiments for dynamic identification and force estimation.

-   `data/`: Contains robot models (`.pkl`), sample motion data, and identification results.
-   `ident/`: Core scripts for identification and estimation.
    -   `ident/real/`: Scripts for experiments with the real robot, including multi-algorithm identification (`ident_multi_algo.py`) and momentum-based force prediction (`fore_calibrate/pred_momentum.py`).
    -   `ident/sim/`: Scripts for simulation, including an interactive demo with a GUI (`sim_interaction_qt.py`).
-   `utils/`: Utility functions, including robot model generation and data processing.

The typical workflow involves:
1.  Running an identification script from `ident/real/` to compute the robot's dynamic parameters from collected data.
2.  Using these parameters in either the simulation (`ident/sim/`) or real-world force estimation scripts.




---

## 🚀 Getting Started

### 🔍 Run Identification
```bash
cd ident/real/
python ident_single.py
```

---

### 🧪 Try in Simulation with UI (**Recommended**)
We deployed our algorithm in the simulation, allowing the selection of force-applied members through an interactive UI, and printed the detection results.
```bash
cd ident/sim/
python sim_interaction_qt.py
```


<div style="text-align:center">
    <img src="https://github.com/user-attachments/assets/72afd041-943a-4011-bf4a-8e8b357d9397" width="100%">
</div>

---

### 📡 Momentum-Based Force Estimation (Real World)

```bash
cd ident/real/fore_calibrate/
python pred_nacfo.py --K 2.5
```

---


## 🙏 Acknowledgements

This project builds upon the work of several amazing open-source libraries, including:

- **[GX11](https://github.com/Democratizing-Dexterous/libgex)**: For dexterous hand hardware.
- **[SymPyBotics](https://github.com/cdsousa/SymPyBotics)**: For robot dynamic identification.
- **[LeRobot](https://github.com/huggingface/lerobot)**: For state-of-the-art imitation learning.



We are grateful to the developers and communities behind these projects.

---
