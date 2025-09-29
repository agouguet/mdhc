# MDHC: Multi-branch Deep Reinforcement-learning for Human-aware Control

**MDHC** is a novel Deep Reinforcement Learning (DRL) approach for **Human-Aware Navigation (HAN)** in mobile robotics.
It introduces a **multi-branch neural network architecture** combined with **Curriculum Learning (CL)** to enable robots to navigate safely, efficiently, and socially in environments shared with humans.

<img src="/figures/sim.gif" height="250" /> <img src="/figures/real.gif" height="250" />   

[[Website]](https://sites.google.com/view/crowdnav-height/home) [[arXiv]](https://arxiv.org/abs/2411.12150) [[Videos]](https://www.youtube.com/playlist?list=PLL4IPhbfiY3ZjXE6wwfg0nffFr_GLtwee)  

This repository provides the implementation of MDHC, along with training scripts, evaluation protocols, and example scenarios.



---

## 🔑 Key Features

* **Multi-branch architecture**
  Separate branches process robot state, environment perception (e.g., lidar), and human-related information before fusion for decision-making.

* **Curriculum Learning**
  Progressive training strategy: from simple to complex scenarios to improve stability and generalization.

* **Human-aware decision-making**
  The agent learns to balance navigation efficiency with socially acceptable behaviors.

* **Integration with RobotSNAP**
  Training and evaluation performed in **RobotSNAP**, a realistic and flexible benchmarking platform for HAN.

---

## 📦 Installation

```bash
git clone https://github.com/agouguet/mdhc.git
cd mdhc
pip install -r requirements.txt
```

Requirements include:

* Python 3.8+
* PyTorch
* ROS
* RobotSNAP

---

## 🚀 Usage

### Training

```bash
python train.py --config configs/mdhc_train.yaml
```

### Evaluation

```bash
python evaluate.py --checkpoint runs/mdhc_best.pth --scenario scenarios/mixed_dense_sparse.json
```

---

## 📊 Results

Extensive simulation in RobotSNAP and preliminary real-world experiments show:

* **MBSN**: interpretable and effective but computationally heavy.
* **MDHC (this repo)**: real-time, adaptive, but requires training and shows robustness challenges in sparse environments.
* **HBSN**: best overall compromise, simple and robust.

MDHC demonstrated strong performance in dense and mixed scenarios, highlighting the benefits of curriculum learning and the multi-branch architecture.

---

## 📂 Repository Structure

```
MDHC/
├── configs/        # Training and evaluation configs
├── models/         # Multi-branch architecture definitions
├── train.py        # Training entry point
├── evaluate.py     # Evaluation script
└── utils/          # Helper functions
```

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@phdthesis{YourName2025,
  title={Human-Aware Robot Navigation: Models, Heuristics, and Deep Reinforcement Learning},
  author={Your Name},
  school={Your University},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a pull request or raise an issue.

---

## 📧 Contact

For questions or collaborations:
**Your Name** – [your.email@domain.com](mailto:your.email@domain.com)
