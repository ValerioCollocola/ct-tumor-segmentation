import nibabel as nib

def load_nifti(file_path):
    # Carica il file NIfTI utilizzando nibabel con percorso relativo
    nifti_img = nib.load(file_path)
    
    # Estrai i dati del volume come array NumPy
    volume = nifti_img.get_fdata()
    
    # Estrai la matrice affine associata all'immagine
    affine = nifti_img.affine
    
    return volume, affine
