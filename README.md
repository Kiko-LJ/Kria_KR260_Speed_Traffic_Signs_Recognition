# Real-Time Speed Traffic Sign Recognition System on AMD/Xilinx Kria KR260

[![Hardware](https://img.shields.io/badge/Hardware-AMD%2fXilinx%20Kria%20KR260-blue)](https://www.xilinx.com/products/som/kria/kr260-robotics-starter-kit.html)
[![Vitis AI](https://img.shields.io/badge/Vitis%20AI-v3.0-green)](https://github.com/Xilinx/Vitis-AI)
[![Model](https://img.shields.io/badge/Model-YOLOv5-orange)](https://github.com/ultralytics/yolov5)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-red.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This repository contains the complete hardware and software implementation of a real-time **Traffic Sign Recognition System** developed for the **AMD/Xilinx Kria KR260 Robotics Starter Kit**. 

The system utilizes a custom **YOLOv5** model trained on speed limit traffic signs, quantized with **Vitis AI 3.0**, and deployed onto a high-performance **DPU CZDX8G (B4096)** core configured within Vivado and running under PetaLinux.

---

## 📌 Features

* **Edge AI Acceleration:** Deployed on DPU B4096 architecture achieving high-throughput real-time inference.
* **Computer Vision Pipeline:** Asynchronous frame capture and display using OpenCV and Python VART (Vitis AI Runtime) APIs.
* **Custom Firmware Overlay:** Dynamically loadable hardware overlay managed via `xmutil`.
* **Multi-Class Detection:** Detects and classifies speed limit signs (20, 30, 40, 50, 60, 70, 80, 90, 100, and 120 km/h).

---

## 📁 Repository Structure

```text
.
├── kr260-dpu-trd-b4096-gpio/   # PetaLinux firmware (.bit, .dtbo, shell.json)
├── inference/                  # Compiled INT8 model (.xmodel) and Python VART inference script
├── vivado_design/              # Complete Vivado project and DPU IP hardware sources
├── assets/                     # Images, diagrams, and benchmarks used in this README
└── README.md                   # Main documentation
