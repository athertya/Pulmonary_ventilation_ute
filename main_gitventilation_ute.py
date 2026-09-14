#!/usr/bin/env python3
"""
Pulmonary Ventilation MRI Mapping (UTE)
Deformable registration via ANTsPy to compute:
  - Regional Ventilation (RV): det(J) - 1.0 with Gaussian smoothing
  - Specific Ventilation (SV): native-scale intensity parenchymal map
"""

import os
import argparse
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import ants


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pulmonary ventilation mapping on sample UTE data."
    )
    parser.add_argument(
        "--input_mat",
        type=str,
        default="gitdata/sample_data.mat",
        help="Path to input .mat file (contains 4D array [X, Y, Z, 2])",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gitdata/results",
        help="Directory to save output .mat and .npy results",
    )
    parser.add_argument(
        "--type_of_transform",
        type=str,
        default="antsRegistrationSyN[b]",
        help="ANTs registration transform type",
    )
    parser.add_argument(
        "--sigma_pre",
        type=float,
        default=1.5,
        help="Gaussian filter sigma applied prior to registration",
    )
    parser.add_argument(
        "--sigma_rv_post",
        type=float,
        default=4.0,
        help="Gaussian filter sigma applied post-hoc to the RV map",
    )
    return parser.parse_args()


def load_input_data(mat_path):
    print(f"Loading dataset: {mat_path}")
    data = sio.loadmat(mat_path)

    # Filter out MATLAB internal metadata keys
    valid_keys = [k for k in data.keys() if not k.startswith("__")]
    if not valid_keys:
        raise ValueError("No valid variables found in the provided .mat file.")

    var_name = valid_keys[0]
    volume = np.asarray(data[var_name], dtype=np.float32)

    # Check 4D shape: [X, Y, Z, 2]
    if volume.ndim != 4 or volume.shape[3] < 2:
        raise ValueError(
            f"Expected a 4D array with at least 2 channels, got shape {volume.shape}"
        )

    # Channel 0: Expiration (Fixed), Channel 1: Inspiration (Moving)
    img_exp = volume[:, :, :, 0]
    img_ins = volume[:, :, :, 1]
    return img_exp, img_ins


def compute_ventilation(img_exp, img_ins, args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Pre-smoothing
    print(f"Applying pre-smoothing with sigma = {args.sigma_pre}...")
    img_exp_sm = gaussian_filter(img_exp, sigma=args.sigma_pre, mode="reflect")
    img_ins_sm = gaussian_filter(img_ins, sigma=args.sigma_pre, mode="reflect")

    # 2. Convert to ANTs images
    fixed_ants = ants.from_numpy(img_exp_sm)
    moving_ants = ants.from_numpy(img_ins_sm)

    # 3. Deformable Image Registration (SyN)
    print(f"Running deformable registration ({args.type_of_transform})...")
    outprefix = os.path.join(args.output_dir, "tmp_ants_")
    reg = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=args.type_of_transform,
        outprefix=outprefix,
    )

    # 4. Compute Jacobian Determinant & Regional Ventilation (RV)
    print("Computing deformation Jacobian and Regional Ventilation (RV)...")
    jac_ants = ants.create_jacobian_determinant_image(
        domain_image=fixed_ants,
        tx=reg["fwdtransforms"][0],
        do_log=False,
        geom=False,
    )
    jac = jac_ants.numpy()

    # RV = det(J) - 1.0 with post-smoothing
    rv_raw = jac - 1.0
    rv = gaussian_filter(rv_raw, sigma=args.sigma_rv_post, mode="reflect")

    # 5. Compute Specific Ventilation (SV)
    print("Computing native-scale Specific Ventilation (SV)...")
    img_ins_reg = reg["warpedmovout"].numpy()
    denom = img_ins_reg.copy()
    denom[np.abs(denom) < 1e-6] = 1e-6
    sv = (img_exp_sm - img_ins_reg) / denom

    # 6. Save Outputs to disk (.mat & .npy)
    print(f"Saving outputs to {args.output_dir}...")
    outputs = {
        "rv_map.mat": {"rv": rv},
        "sv_map.mat": {"sv": sv},
        "jac_map.mat": {"jac": jac},
        "img_ins_registered.mat": {"img_ins_reg": img_ins_reg},
        "img_exp_smooth.mat": {"img_exp_smooth": img_exp_sm},
        "img_ins_smooth.mat": {"img_ins_smooth": img_ins_sm},
    }

    for filename, mat_dict in outputs.items():
        sio.savemat(os.path.join(args.output_dir, filename), mat_dict)

    # Also save RV and SV as raw numpy binaries
    np.save(os.path.join(args.output_dir, "rv_map.npy"), rv)
    np.save(os.path.join(args.output_dir, "sv_map.npy"), sv)

    # Clean up temporary ANTs registration transform files
    for f in os.listdir(args.output_dir):
        if f.startswith("tmp_ants_"):
            try:
                os.remove(os.path.join(args.output_dir, f))
            except OSError:
                pass

    print("Ventilation mapping complete.")


def main():
    args = parse_args()
    img_exp, img_ins = load_input_data(args.input_mat)
    compute_ventilation(img_exp, img_ins, args)


if __name__ == "__main__":
    main()