import numpy as np
import nibabel as nib
import subprocess
import os
import torch
import tempfile

def acquire_logits(img, affine):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create temporary file names
        temp_input = os.path.join(tmpdirname, 'input.nii.gz')
        temp_output = os.path.join(tmpdirname, 'output.nii.gz')
        temp_logits = os.path.join(tmpdirname, 'logits.pt')

        # Save input image to temporary file
        nib.save(nib.Nifti1Image(img, affine), temp_input)

        subprocess.run([
            'python', 'mri_synthstrip.py',
            '-i', temp_input,
            '-o', temp_output,
            '--save-logits', temp_logits
        ], check=True)

        logits_tensor = torch.load(temp_logits, weights_only=True)
        print(logits_tensor.shape)

        logits_np = logits_tensor.numpy()

    logits_np = logits_np.squeeze()
    logits_np = logits_np.squeeze()
    return logits_np


def preprocess_image_min_max(img: np.ndarray):
    "Min max scaling preprocessing for the range 0..1"
    img = (img - img.min()) / (img.max() - img.min())
    return img


def preprocessing_pipe(img, lab, affine):
    """ Set up your preprocessing options here, ignore if none are needed """
    img = img
    #img = data
    #img_nifti = nib.Nifti1Image(img, affine=affine)

    #nifti_path = 'img.nii.gz'
    #nib.save(img_nifti, nifti_path)
    #image preprocessing steps
    img = preprocess_image_min_max(img) * 255
    img = img.astype(np.uint8)

    #lab preprocessing steps
    lab = acquire_logits(img, affine)
    print(img)
    print(lab)
    print(lab)
    print(lab.shape)
    lab = lab.astype(np.uint8)
    return (img, lab)
