# Pulmonary Ventilation MRI Mapping (UTE)

An automated image registration and functional lung ventilation pipeline for multi-phase pulmonary MRI. It uses deformable image registration via **ANTsPy** to calculate:
* **Regional Ventilation (RV):** Derived from the forward transformation Jacobian determinant ($RV = \det(J) - 1.0$)
* **Specific Ventilation (SV):** Intensity-based parenchyma ventilation normalized across respiratory states.

Outputs are exported as `.mat` files formatted for inspection in (MATLAB) alongside raw and smoothed references.

---

## 📁 Repository Structure

```text
Pulmonary_ventilation_ute/
├── gitdata/                     # Sample input data and output results
│   ├── img_Jj_2025_12_17.mat # 4D MAT input volume [X, Y, Z, 4]
│   └── nex2_results/            # Generated outputs (MAT & NPY files)
├── main_gitventilation_ute.py   # Primary Python registration & mapping pipeline
├── output_read_mat.m            # MATLAB script to inspect all outputs in MatrixUser
├── requirements.txt             # Python dependencies
└── README.md

Markdown## ⚙️ Requirements

Install dependencies directly using pip:

```bash
pip install antspyx numpy scipy
🚀 Usage1. Run Ventilation Mapping (Python)To run the pipeline on the sample NEX=2 dataset using default parameters:Bashpython main_gitventilation_ute.py --input_mat gitdata/img_Jj_2025_12_17.mat --output_dir gitdata/nex2_results
Input 4D Format Specification:The input .mat file must store a 4D array [X, Y, Z, 4] corresponding to:Index 0: Expiration (NEX = 1)Index 1: Inspiration (NEX = 1)Index 2: Expiration (NEX = 2) (Fixed reference for NEX=2)Index 3: Inspiration (NEX = 2) (Moving volume for NEX=2)Output Files:rv_map.mat (Key: 'rv', Gaussian post-smoothed $\sigma=4.0$)sv_map.mat (Key: 'sv', native intensity scale)jac_map.mat (Key: 'jac')img_ins_registered.mat (Key: 'img_ins_reg')img_exp_smooth.mat (Key: 'img_exp_smooth')img_ins_smooth.mat (Key: 'img_ins_smooth')2. Visualize in MatrixUser (MATLAB)Open MATLAB, navigate to this directory, and run:Matlaboutput_read_mat
This compiles a 7-channel 4D volume (img_check) and opens it in MatrixUser:Channel 1: Raw Moving (Inspiration)Channel 2: Raw Fixed (Expiration)Channel 3: Smoothed MovingChannel 4: Smoothed FixedChannel 5: Registered Moving ([0, 200] display window)Channel 6: Specific Ventilation (sv) ([-0.3, 0.2] display window, Colormap: Jet)Channel 7: Regional Ventilation (rv) ([-0.1, 0.3] display window, Colormap: Jet)

## 📥 Sample Dataset

Due to GitHub's file size limit (100 MB), the sample 4D UTE lung MRI dataset (`img_Jj_2025_12_17.mat`, ~115 MB) is hosted openly on **Zenodo**:

* **Zenodo Record:** [https://zenodo.org/records/22758690](https://zenodo.org/records/22758690)

#### Download & Setup:
1. Download `img_Jj_2025_12_17.mat` from Zenodo.
2. Place the file inside the local `gitdata/` directory:
   ```text
   Pulmonary_ventilation_ute/
   └── gitdata/
       └── img_Jj_2025_12_17.mat