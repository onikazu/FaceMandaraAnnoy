# FaceMandara
## Abstract
FaceMandara searches similar face to you in real time from over 200,000 face data(celebA).

## Dependency(I checked it could work)
OS: MacOS10.13.6 or Ubuntu16.04

dlib

face_recognition

faiss

opencv-python

## Usage
- At first, install Anaconda.

- make virtual env by using `conda`
~~~
$ conda create -n face_mandara python=3.5 dlib opencv-python face_recognition numpy
$ source activate face_mandara
~~~

- install faiss which is the package which can do Product Quantization from FaceBook
```
# CPU version only
$ conda install faiss-cpu -c pytorch
# Make sure you have CUDA installed before installing faiss-gpu, otherwise it falls back to CPU version
$ conda install faiss-gpu -c pytorch # [DEFAULT]For CUDA8.0
$ conda install faiss-gpu cuda90 -c pytorch # For CUDA9.0
$ conda install faiss-gpu cuda91 -c pytorch # For CUDA9.1
# cuda90/cuda91 shown above is a feature, it doesn't install CUDA for you.
```

- download CelebA dataset and put the data to "big_database_def" directory.
[CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

- Rename the directory "big_database_def" to "big_database"

- Then put the command terminal like bellow

```
$ python image2vector.py
$ python face_mandara.py
```

- You don't have to do this command from next execution
```
$ python image2vector.py
```
