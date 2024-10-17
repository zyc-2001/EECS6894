# Zero-Shot-test
The repo is a simple test of Zero-Shot-NAS based on the approach from https://github.com/SLDGroup/survey-zero-shot-nas. The symbols '%' and '!' in some parts of the code were added to run directly in Colab, and can be removed when using other bash environments.
## 1. Create a working directory

To ensure that files and data can be correctly stored and accessed, I created a `/home/test0` directory and switched to this directory. All subsequent file operations will be conducted in this directory.

```bash
!mkdir /home/test0
%cd /home/test0
!pwd
```

## 2. Clone the nasbench repository and install it

This step clones the NAS-Bench repository and installs the necessary dependencies. Since NAS-Bench uses an older version of TensorFlow, I replaced the original TensorFlow in the code with `tensorflow.compat.v1` to ensure compatibility in the current environment. Afterward, I installed the repository using `pip install -e .`.

```bash
!git clone https://github.com/google-research/nasbench /home/test0/nasbench
%cd /home/test0/nasbench
!sed -i 's/import tensorflow as tf/import tensorflow.compat.v1 as tf/g' nasbench/api.py
!sed -i 's/import tensorflow as tf/import tensorflow.compat.v1 as tf/g' nasbench/lib/evaluate.py
!sed -i 's/import tensorflow as tf/import tensorflow.compat.v1 as tf/g' nasbench/lib/training_time.py
!pip install -e .
```

## 3. Download NAS-Bench datasets

This step downloads the NAS-Bench datasets, specifically NASBench-101 and NASBench-201. Some dataset download links have expired, such as ImageNet16-120. Therefore, I used CIFAR-10 and CIFAR-100 datasets and downloaded the NAS-Bench-201 dataset from a Google Drive link using `gdown`.

```bash
!mkdir -p /home/test0/dataset/nasbench/
%cd /home/test0/dataset/nasbench/
!wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord  # NASBench-101
!gdown https://drive.google.com/uc?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_  # NASBench-201
!mkdir -p /home/test0/dataset/nasbench/NATS/
%cd /home/test0/dataset/nasbench/NATS/
!gdown https://drive.google.com/uc?id=1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA
```

## 4. Clone the zero-shot NAS project

To evaluate the architectures, I cloned the survey-zero-shot-nas project. This project implements several zero-shot proxies for architecture evaluation. I will later adjust the code and use it for architecture evaluation.

```bash
%cd /home/test0/
!git clone https://github.com/SLDGroup/survey-zero-shot-nas /home/test0/survey-zero-shot-nas
%cd /home/test0/survey-zero-shot-nas
```

## 5. Install the required dependencies

To ensure that the NAS-Bench API works properly in the project, I installed the necessary libraries, including nats_bench, NAS-Bench-201, and AutoDL-Projects dependencies. I also installed `ptflops` to calculate the model's floating point operations (FLOPs) and parameter counts.

```bash
!pip3 install nats_bench
!git clone https://github.com/D-X-Y/NAS-Bench-201.git
%cd NAS-Bench-201
!pip3 install -e .
!git clone https://github.com/D-X-Y/AutoDL-Projects.git
%cd AutoDL-Projects
!pip3 install -e .
!pip3 install ptflops
```

## 6. Run the main program

Once everything was ready, I ran `main.py` to evaluate the architectures in the NAS-Bench-101 search space. I used the CIFAR-10 dataset and selected "basic" as the zero-shot proxy to evaluate the architectures. The evaluation results were saved in a CSV file.

```bash
%cd /home/test0/survey-zero-shot-nas
!python3 main.py --searchspace=101 --dataset=cifar10 --data_path ~/dataset/ --metric=basic
```

## 7. Issues discovered during the run

### 7.1 Circular import
During the run, we encountered several issues. First, there was a circular import problem, specifically between `__init__.py` and other modules. To fix this, I removed the direct imports of `grad_norm`, `snip`, and `grasp` from the top of the `__init__.py` file and replaced them with a `lazy_import` function to load these modules only when necessary. This resolved all circular import issues. The updated file, `__init__-m.py`, has been uploaded.

![image](https://github.com/user-attachments/assets/bd4e9ef6-1504-49cd-8e13-88a38a601365)


### 7.2 Formatting errors
Another issue was with `main.py`, which had some formatting errors. The fixed version, `main-m.py`, has been uploaded as well. To save time, I also added a feature in `main-1000.py` that only processes the first 1000 network architectures and calculates the total runtime.

## 8. Results
The final result on Colab's T4 GPU was that 1000 network architectures took about 425 seconds to evaluate, which met the expected performance. The specific results will be compared with other evaluation models and updated later.
