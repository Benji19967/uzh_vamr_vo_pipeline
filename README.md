# Visual Odometry Pipeline for the course VAMR at UZH

## Demo

![Demo](https://private-user-images.githubusercontent.com/35387157/566976049-3ca81489-7267-4a11-8883-65c4121c7038.mov?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzQwMjcyNjMsIm5iZiI6MTc3NDAyNjk2MywicGF0aCI6Ii8zNTM4NzE1Ny81NjY5NzYwNDktM2NhODE0ODktNzI2Ny00YTExLTg4ODMtNjVjNDEyMWM3MDM4Lm1vdj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzIwVDE3MTYwM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWViZTg2ZGY0ZjkzOTk5OWQyYTZjNWZjYjA5YjA5MjA4MDk4MjNhYjNhNjcyNzQ2ODJjMWFjZmY5MDU2MjIwNWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.iOZzwjP7txKYX3eSng4V0M4c0ZTqLgauO2yD8DqECiA)

## Getting started

1. Install the requirements in a Python venv
2. Activate the environment
3. `python src/main.py`
4. Download the parking dataset:
  - `make data/parking`


## Using the CLI

```bash
vo --help
vo run --dataset parking
```

## Conventions

Points are stored as __column vectors__ to facilitate linear algebra matrix multiplications.

__Notation__: 

- `p_W`: points in the World coordinate frame
- `p_I`: points in the Image coordinate frame 
- `p_C`: points in the Camera coordinate frame
- `_hom`: homogenous points
- `T_C_W`: transformation from the World to the Camera coordinate frame. 

__Example__: 

The homogenous 3D points (1, 2, 3, 1) and (4, 5, 6, 1) in the World frame:
```
p_W_hom = np.ndarray(
    [1, 4],
    [2, 5],
    [3, 6],
    [1, 1],
)
```

## Notable progress

### Version 0

1. Adding keypoints using grids. Avoids clustering of keypoints and distributes them over the entire image.
2. Limit number of candidate keypoints. 
    - Eliminates old candidate keypoints (__after__ new ones were added) which may be out of frame and leads to less drift.
    - More efficient computationally
    - Interesting point: returning the new candidate keypoints in random order (shuffling) seems to increase drift.
    
### Version 1

1. Use images 0 and 4 (rather than 0 and 2) for initialization. This reduced the overall scale drift significantly because the initialization of the 3D point cloud is more accurate. 
2. Use 4 degrees instead of 5 degrees as the min angle to triangulate. 

### Version 2

1. Apply PNP pose refinement on RANSAC inliers

### Version 3

1. Use 6.0 pixels as a filter in PNP RANSAC rather than 8.0 

## Demo (takes a few seconds to load)

- There is __scale drift__ because this is a monocular setup. 
- Demo of version "0"

![Parking VO](assets/00/parking_00.gif)

## Evaluation

- Evaluation of version "3": there is less scale drift. 

![Camera Trajectory](assets/03/camera_trajectory.png)

![Reprojection Error](assets/03/reprojection_error.png)

![Scale Drift](assets/03/scale_drift.png)
