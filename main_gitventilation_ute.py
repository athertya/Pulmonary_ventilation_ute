#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pulmonary Regional & Specific Ventilation Mapping Pipeline 
====================================================================

Description:
    Processes dual-phase 3D pulmonary MRI volumes (Expiration and Inspiration)
    acquired to derive biomechanical and functional
    ventilation biomarkers:
      1. Regional Ventilation (RV):
         Derived from the determinant of the deformation Jacobian tensor
         evaluating the forward coordinate transformation (Warp moving -> fixed):
             RV = det(J) - 1.0
      2. Specific Ventilation (SV):
         Derived from the parenchymal signal intensity drop across phases,
         normalized to native signal scale:
             SV = (I_exp_smooth - I_reg_smooth) / (I_reg_smooth + eps)

Pipeline Stages:
    - Load 4D MAT array [X, Y, Z, 4] or dedicated Expiration/Inspiration pairs.
    - Extract NEX = 2 phase indices (Index 2: Expiration, Index 3: Inspiration).
    - Gaussian pre-smoothing on normalized arrays (sigma = 1.5) to stabilize optimization.
    - Deformable symmetric normalization registration via ANTsPy (SyN).
    - Forward-warp Jacobian determinant extraction (det(J)).
    - Direct RV evaluation (RV = J - 1.0) with Gaussian post-smoothing (sigma = 4.0).
    - Native-scale SV calculation with reflection boundary Gaussian smoothing.
    - Export MatrixUser-ready .mat files and NumPy .npy arrays.

Author:
    UTE Lab
"""

import os
import glob
import argparse
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage
import ants


def smooth_volume(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Apply isotropic 3D Gaussian reflection smoothing to a volume.

    Parameters:
        img (np.ndarray): 3D input image array.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: Smoothed 3D array.
    """
    return ndimage.gaussian_filter(img, sigma=sigma, mode='reflect', truncate=2.0)


def calculate_regional_ventilation(jac: np.ndarray) -> np.ndarray:
    """
    Calculate direct Regional Ventilation (RV) map from the Jacobian determinant.

    In pulmonary biomechanics, local tissue volumetric expansion relative to the
    end-expiration baseline is given by:
        RV = det(J) - 1.0

    Parameters:
        jac (np.ndarray): 3D Jacobian determinant map (det(J)).

    Returns:
        np.ndarray: Regional ventilation map.
    """
    return jac - 1.0


def calculate_specific_ventilation(img_exp: np.ndarray, img_ins_reg: np.ndarray) -> np.ndarray:
    """
    Calculate Specific Ventilation (SV) using the signal intensity drop model:
        SV = (I_exp - I_reg) / (I_reg + eps)

    Parameters:
        img_exp (np.ndarray): End-expiration volume in native signal intensity scale.
        img_ins_reg (np.ndarray): Deformably registered end-inspiration volume
                                  in native signal intensity scale.

    Returns:
        np.ndarray: Specific ventilation map.
    """
    eps = np.finfo(float).eps
    return (img_exp - img_ins_reg) / (img_ins_reg + eps)


def run_ventilation_pipeline(
    input_mat: str,
    output_dir: str,
    trans: str = 'antsRegistrationSyN[b]',
    sigma_pre: float = 1.5,
    sigma_rv_post: float = 4.0
):
    """
   

    Parameters:
        input_mat (str): Path to input .mat file containing 4D volume data.
        output_dir (str): Directory where output maps will be stored.
        trans (str): ANTs registration transformation model.
        sigma_pre (float): Gaussian filter sigma applied before registration.
        sigma_rv_post (float): Gaussian filter sigma applied to final RV map.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PULMONARY VENTILATION PIPELINE")
    print("=" * 70)
    print(f"Input file      : {input_mat}")
    print(f"Output directory: output/ ")
    print(f"Transform type  : {trans}")
    print(f"Pre-smooth sigma: {sigma_pre}")
    print(f"RV post-smooth  : {sigma_rv_post}")
    print("-" * 70)

    # -------------------------------------------------------------------------
    # 1. Load MAT File and Extract NEX = 2 Pair
    # -------------------------------------------------------------------------
    print("[Step 1/6] Loading 4D MAT volume...")
    mat_contents = sio.loadmat(input_mat)
    data_keys = [k for k in mat_contents.keys() if not k.startswith('__')]
    if not data_keys:
        raise KeyError(f"No valid data array found in {input_mat}")

    volume_4d = np.abs(np.squeeze(mat_contents[data_keys[0]])).astype(np.float32)

    # Validate dimensionality: Expecting [X, Y, Z, 4]
    # Index 0: Expiration NEX=1 | Index 1: Inspiration NEX=1
    # Index 2: Expiration NEX=2 | Index 3: Inspiration NEX=2
    if volume_4d.ndim != 4 or volume_4d.shape[-1] < 4:
        raise ValueError(
            f"Expected 4D array with at least 4 volumes along Dim 4, got shape {volume_4d.shape}"
        )

    print("Extracting NEX = 2 respiratory phases (Index 2: EE, Index 3: EI)...")
    img_exp_raw = volume_4d[..., 2]  # End-Expiration (Fixed Reference)
    img_ins_raw = volume_4d[..., 3]  # End-Inspiration (Moving Floating)

    orig_max_exp = float(np.max(img_exp_raw))
    orig_max_ins = float(np.max(img_ins_raw))

    # -------------------------------------------------------------------------
    # 2. Intensity Normalization & Gaussian Pre-Smoothing
    # -------------------------------------------------------------------------
    print("[Step 2/6] Normalizing intensities and pre-smoothing input volumes...")
    img_exp_norm = img_exp_raw / orig_max_exp
    img_ins_norm = img_ins_raw / orig_max_ins

    # Pre-smoothing reduces high-frequency MRI noise and prevents local minima
    img_exp_smooth = smooth_volume(img_exp_norm, sigma=sigma_pre)
    img_ins_smooth = smooth_volume(img_ins_norm, sigma=sigma_pre)

    # -------------------------------------------------------------------------
    # 3. ANTs Deformable Symmetric Normalization (SyN) Registration
    # -------------------------------------------------------------------------
    print(f"[Step 3/6] Running ANTs deformable registration ({trans})...")
    fix_ants = ants.from_numpy(img_exp_smooth)
    mov_ants = ants.from_numpy(img_ins_smooth)

    tmp_prefix = os.path.join(output_dir, f"tmp_ants_{np.random.randint(0, 10000)}_")

    reg_output = ants.registration(
        fixed=fix_ants,
        moving=mov_ants,
        type_of_transform=trans,
        initial_transform=None,
        grad_step=0.1,
        flow_sigma=3.0,
        total_sigma=1.0,
        aff_metric='mattes',
        syn_metric='cc',
        syn_sampling=32,
        reg_iterations=(40, 20, 10),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
        outprefix=tmp_prefix
    )

    # Extract warped moving image and apply consistent spatial smoothing
    img_ins_reg_norm = reg_output['warpedmovout'].numpy()
    img_ins_reg_smooth = smooth_volume(img_ins_reg_norm, sigma=sigma_pre)

    # -------------------------------------------------------------------------
    # 4. Compute Jacobian Determinant & Regional Ventilation (RV) Map
    # -------------------------------------------------------------------------
    print("[Step 4/6] Computing forward-warp Jacobian determinant and RV map...")
    # Forward transform evaluates expansion of inspiration relative to expiration
    jac_ants = ants.create_jacobian_determinant_image(
        domain_image=fix_ants,
        tx=reg_output['fwdtransforms'][0],
        do_log=False
    )
    jac_raw = jac_ants.numpy()

    # Direct RV evaluation: det(J) - 1.0
    rv_raw = calculate_regional_ventilation(jac_raw)

    # Apply Gaussian smoothing with sigma = 4.0 to regularize the final RV map
    print(f"Applying Gaussian smoothing (sigma={sigma_rv_post}) to final RV map...")
    rv = smooth_volume(rv_raw, sigma=sigma_rv_post)

    # -------------------------------------------------------------------------
    # 5. Compute Specific Ventilation (SV) Map in Native Intensity Scale
    # -------------------------------------------------------------------------
    print("[Step 5/6] Computing Specific Ventilation (SV) map in native intensity scale...")
    # Rescale smoothed volumes back to native signal levels prior to SV evaluation
    img_exp_native = img_exp_smooth * orig_max_exp
    img_ins_reg_native = img_ins_reg_smooth * orig_max_ins

    sv_raw = calculate_specific_ventilation(img_exp_native, img_ins_reg_native)
    # SV keeps default baseline smoothing (sigma = 1.0)
    sv = smooth_volume(sv_raw, sigma=1.0)

    # Rescale registered volume to native intensity scale for visualization
    img_ins_reg_scaled = img_ins_reg_norm * orig_max_ins

    # Clean up temporary ANTs transform files from disk
    for tmp_file in glob.glob(tmp_prefix + '*'):
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    # -------------------------------------------------------------------------
    # 6. Save MatrixUser-Ready MAT Files & NumPy Arrays
    # -------------------------------------------------------------------------
    print("[Step 6/6] Exporting results to disk...")

    # Save MATLAB MAT files (.mat)
    sio.savemat(os.path.join(output_dir, 'rv_map.mat'), {'rv': rv})
    sio.savemat(os.path.join(output_dir, 'sv_map.mat'), {'sv': sv})
    sio.savemat(os.path.join(output_dir, 'jac_map.mat'), {'jac': jac_raw})
    sio.savemat(os.path.join(output_dir, 'img_ins_registered.mat'), {'img_ins_reg': img_ins_reg_scaled})
    sio.savemat(os.path.join(output_dir, 'img_exp_smooth.mat'), {'img_exp_smooth': img_exp_native})
    sio.savemat(os.path.join(output_dir, 'img_ins_smooth.mat'), {'img_ins_smooth': img_ins_reg_native})

    # Save NumPy array backups (.npy)
    np.save(os.path.join(output_dir, 'rv_map.npy'), rv)
    np.save(os.path.join(output_dir, 'sv_map.npy'), sv)
    np.save(os.path.join(output_dir, 'jac_map.npy'), jac_raw)
    np.save(os.path.join(output_dir, 'ins_registered.npy'), img_ins_reg_scaled)

    print("-" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Outputs successfully saved to: {output_dir}")
    print("Files available for MatrixUser:")
    print("  - rv_map.mat             (Key: 'rv', Gaussian post-smoothed sigma=4.0)")
    print("  - sv_map.mat             (Key: 'sv', calculated in native intensity scale)")
    print("  - jac_map.mat            (Key: 'jac')")
    print("  - img_ins_registered.mat (Key: 'img_ins_reg')")
    print("  - img_exp_smooth.mat     (Key: 'img_exp_smooth')")
    print("  - img_ins_smooth.mat     (Key: 'img_ins_smooth')")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pulmonary Regional (RV) and Specific Ventilation (SV) Mapping '
    )
    parser.add_argument(
        '--input_mat',
        type=str,
        default='gitdata/img_Jj_2025_12_17.mat',
        help='Path to the 4D input MAT file containing the respiratory phases'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results_Jj_NEX2_forwardTf',
        help='Directory to store output registration volumes and ventilation maps'
    )
    parser.add_argument(
        '--type_of_transform',
        type=str,
        default='antsRegistrationSyN[b]',
        help='ANTs transformation model (default: antsRegistrationSyN[b])'
    )
    parser.add_argument(
        '--sigma_pre',
        type=float,
        default=1.5,
        help='Gaussian pre-smoothing standard deviation in voxels (default: 1.5)'
    )
    parser.add_argument(
        '--sigma_rv_post',
        type=float,
        default=4.0,
        help='Gaussian post-smoothing standard deviation for the RV map (default: 4.0)'
    )

    args = parser.parse_args()

    run_ventilation_pipeline(
        input_mat=args.input_mat,
        output_dir=args.output_dir,
        trans=args.type_of_transform,
        sigma_pre=args.sigma_pre,
        sigma_rv_post=args.sigma_rv_post
    )