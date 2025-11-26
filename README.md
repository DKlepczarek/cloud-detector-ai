# Cloud Detector AI: Deep Learning for Offshore Safety

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![Status](https://img.shields.io/badge/Status-Fine--Tuned-success)
![License](https://img.shields.io/badge/License-MIT-green)

> **"At sea, recognizing a Cumulonimbus 15 minutes earlier is the difference between a smooth sail and a broken mast."**

## 📖 Overview
**Cloud Detector AI** is a computer vision system designed to assist offshore sailors in identifying hazardous weather formations in real-time. By leveraging **Transfer Learning** and a custom **Risk-Aware Decision Engine**, the model classifies cloud types from images and provides actionable sailing advice (e.g., "Reef the sails").

This project bridges the gap between **Quantitative Methods** and **Nautical Science**, addressing the critical problem of cognitive overload and human error during long offshore passages.

---

## Problem
In offshore sailing, weather conditions change rapidly. Less experienced crew members often struggle to distinguish between benign clouds (e.g., *Cumulus*) and dangerous storm precursors (e.g., *Cumulonimbus*), especially under fatigue.

* **Misclassification Risk:** Mistaking a storm cloud for a fair-weather cloud leads to delayed reaction.
* **Consequence:** Sudden squalls can cause equipment failure or capsize if sails are not reduced in time.

## Solution
Cloud Atlas AI acts as a "Second Officer," analyzing the sky continuously.
It is built on **MobileNetV2**, a lightweight Convolutional Neural Network (CNN) optimized for edge devices (like a Raspberry Pi on a yacht), and fine-tuned on a meteorological dataset.

### Key Features
* **Real-Time Classification:** Identifies 11 cloud types (including *Altocumulus, Cumulonimbus, Cirrus*).
* **Safety-First Logic:** Implements a custom decision threshold to prioritize **Recall** over Precision for dangerous classes.
* **Edge-Ready:** Containerized with Docker for easy deployment on onboard systems.

---

## Model Architecture & Methodology

### 1. Transfer Learning & Fine-Tuning
Instead of training from scratch, I utilized **MobileNetV2** pre-trained on ImageNet.
* **Base:** Frozen layers to extract low-level features (edges, textures).
* **Head:** Custom dense layers for cloud classification.
* **Fine-Tuning:** Unfroze the top 100 layers of the base model and retrained them with a low learning rate (`1e-4`) to adapt the model to specific cloud textures.

### 2. Data Augmentation
To simulate the unstable environment of a moving yacht, the training pipeline includes robust augmentation:
* Rotation (`±20°`) to account for heel (przechył jachtu).
* Zoom and Horizontal Flip.

### 3. 🛡️ Safety-Critical Logic (Threshold Tuning)
Standard models maximize accuracy. In safety-critical systems, **Recall** is king.
I implemented a custom inference logic:
```python
# Pseudo-code of the safety logic
if probability(Cumulonimbus) > 0.25:
    return "DANGER: Storm Imminent (Safety Override)"
else:
    return argmax(prediction)

## ⚖️ License & Acknowledgments

* **Code:** Released under the [MIT License](LICENSE). You are free to use, modify, and distribute this software.
* **Data:** The dataset used for training/fine-tuning is sourced from the [CCSN Cloud Database](https://www.kaggle.com/) (or specific Kaggle link). The images belong to their respective authors and are used here solely for **educational and non-commercial purposes**.
* **Disclaimer:** This tool is intended for assistance only. Always rely on official weather forecasts and your own judgment at sea.
