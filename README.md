# 🚦 Real-Time Speed Traffic Sign Recognition System on AMD/Xilinx Kria KR260

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
├── vivado_design/              # Vivado project documentation & external download link
│   └── README.md               # Instructions and link to download heavy Vivado sources
├── assets/                     # Images, diagrams, and benchmarks used in this README
└── README.md                   # Main documentation
```

### Folder Breakdown:
1. **`kr260-dpu-trd-b4096-gpio/`**: Contains the generated hardware overlay files required by PetaLinux to configure the FPGA logic on the fly.
2. **`inference/`**: Houses the final compiled `yolov5_kr260.xmodel` file along with the Python inference script (`inferencia_gpio_mod.py`) for video processing and bounding box rendering.
3. **`vivado_design/`**: Contains a `README.md` with an external download link for the full Vivado project workspace (`prj/`), DPU IP core (`dpu_ip/`), and sample apps (`app/`), omitted directly from Git due to file size constraints.
4. **`assets/`**: Images, system architecture diagrams, and demonstration screenshots.

---

## 🛠️ Hardware & Software Requirements

### Hardware
* **Board:** AMD/Xilinx Kria KR260 Robotics Starter Kit
* **Camera:** USB Webcam (UVC compliant)
* **Display:** Monitor connected via DisplayPort
* **Connections:** Micro-USB cable (for UART serial console) & 12V Power Adapter

### Software & Environment
* **OS:** PetaLinux (custom build for Kria KR260)
* **AI Framework:** Vitis AI 3.0 (VART Python API)
* **Libraries:** OpenCV, NumPy

---

## 🚀 Quick Start Guide

### 1. Hardware Setup
1. Insert the MicroSD card loaded with the PetaLinux image into the Kria KR260.
2. Connect the **USB Webcam** to one of the USB 3.0 ports.
3. Connect the **Monitor** to the DisplayPort connector.
4. Connect the **Micro-USB cable** from the Kria UART port to your Host PC.
5. Power on the board.

### 2. Connect via UART Serial Terminal
Open a serial terminal on your host machine (e.g., PuTTY or `picocom`) at **115200 baud rate**:

```bash
picocom -b 115200 /dev/ttyUSB1
```

Log in using credentials:
* **Username:** `petalinux`
* **Password:** *(Set or enter your password)*

### 3. Copy and Load Firmware Overlay
Transfer the `kr260-dpu-trd-b4096-gpio` folder to `/lib/firmware/xilinx/` on the board.

Unload any active app and load the custom DPU firmware:

```bash
# Unload current app
sudo xmutil unloadapp

# Load the DPU firmware overlay
sudo xmutil loadapp kr260-dpu-trd-b4096-gpio
```

### 4. Run Real-Time Inference
Navigate to the `inference/` folder and execute the Python script:

```bash
cd /path/to/inference
sudo python3 inferencia_gpio_mod.py
```

*The real-time video feed with bounding box detections will be displayed on the screen connected via DisplayPort.*

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

You are free to share and adapt the material for non-commercial, academic, or research purposes, provided appropriate credit is given. **Commercial use is strictly prohibited without explicit permission from the author.**
