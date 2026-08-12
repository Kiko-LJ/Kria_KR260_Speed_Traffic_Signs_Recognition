# 🛠️ Vivado Design & DPU Hardware Files

Due to GitHub's file size limits, the complete Vivado project workspace, IP cores, and heavy synthesis outputs are hosted externally.

---

## 📥 Download Hardware Sources

You can download the full hardware package from the following link:

👉 **[Download Complete Vivado Project & DPU IP Core (External Link)](https://drive.google.com/file/d/1HyJ6v7gbpCEXp-6QvY0FyGg5NflbDd6T/view?usp=sharing)**

---

## 📂 Package Contents

Once downloaded and extracted, the package contains the following structure:

```text
vivado_design/
├── app/        # Sample application reference files (e.g., resnet50)
├── dpu_ip/     # Xilinx DPU IP core source files and repository
└── prj/        # Full Vivado project workspace (.xpr, block designs, IP integrators)
```

## ⚙️ Instructions

1. Download the `.zip` archive from the link above.
2. Extract the contents directly into this `vivado_design/` directory.
3. Open the `.xpr` project file located in `vivado_design/prj/` using **Vivado 2022.2** (or the compatible version used in this project).