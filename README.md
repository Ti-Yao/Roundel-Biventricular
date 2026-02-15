# Roundel

Short-Axis Segementation App.


[Demo](Roundel_Demo.mp4)




### Data Structure

data/
└── {subfolder_name}/     
    ├── image___{pat_id}.nii.gz
    ├── mask___{pat_id}.nii.gz
    └── sax_df___{pat_id}.csv


### File Descriptions

image___{pat_id}.nii.gz  
NIfTI image file for a single patient.  
Data shape: (H, W, D, T), where  
H = height, W = width, D = slice index, T = time frame.

mask___{pat_id}.nii.gz  
Segmentation mask corresponding to the image.  
Data shape: (H, W, D, T).  
Spatial and temporal dimensions match the image exactly.

saxdf___{pat_id}.csv  
CSV file containing acquisition metadata.


### saxdf CSV Structure

Columns:
- pixelspacing
- thickness

Example content:
pixelspacing,thickness  
1,8
