<p float="left">
  <img src="docs/im0.jpg" width="250" />
  <img src="docs/im1.jpg" width="250" /> 
  <img src="docs/overlay_image.png" width="250" />
</p>

Given a pair of images, the algorithm computes the translation (dx, dy), scale and rotation required to register/align one image to the other.
A brief explanation of the algorithm is provided [here](https://sthoduka.github.io/imreg_fmt/docs/overall-pipeline/).

This project is a partial port of the Python implementation by Christoph Gohlke and Matěj Týč (see [here](https://github.com/matejak/imreg_dft)).
It is written in C++ and is suited for registering a sequence of images (such as from a video).
For images of size 320x240, the algorithm runs at approximately 14 Hz on an Intel Core i3 (1.7 GHz).

## Dependencies
* fftw3
* OpenCV 4.2
* Eigen
* Ceres

## Compile
```bash
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
```
### Mac OS X
  `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-isystem /usr/local/include' ..`

## Run
```bash
./image_main <path to load images> output/<path to save results>
```
the input directory structure are like this
```
- <data_dir>
  - Images (this is a directory)
  - cam_param.yaml (the format is like the below part)
  - imageList.txt (each line is the name of image in Images. you can generate this file with `ls Images > imageList.txt` in <data_dir>)
  - timestamps.txt (the content of this is not necessary)
```
cam_param.yaml
```yaml
%YAML:1.0

Camera.type: "PinHole"

Camera.fx: 2955.7
Camera.fy: 2949.9
Camera.cx: 2023.1
Camera.cy: 1510.7

Camera.k1: 0.1601
Camera.k2: -0.3277
Camera.p1: -0.00073991
Camera.p2: 0.0051

Camera.width: 4056
Camera.height: 3040
```
Noted, there should be a directory `txt` under `output`.


License
-------
This project is licensed under the [GPLv3 License](LICENSE). The license of the original Python version by Gohlke and Týč can be found [here](LICENSE-ORIGINAL).

References
----------
[1] B. S. Reddy and B. N. Chatterji, “An FFT-based Technique for Translation,
Rotation, and Scale-Invariant Image Registration,” IEEE Transactions on Image Processing, vol. 5, no. 8, pp. 1266–1271, 1996.
