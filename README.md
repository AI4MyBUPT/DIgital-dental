## Introduction
Placeholder

## How to install

Please use the following command for installation and running.

```bash
# Create a new environment
conda create -n aagt python==3.8
conda activate aagt

# Install packages and other dependencies
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
python setup.py build develop

cd aagt/extensions/pointops/
pip install -v . --no-build-isolation
```

## Demo
We have provided a demo test. After completing the above installation steps, you can run the demo using the following commands:
```bash
cd aagt/experiment
CUDA_VISIBLE_DEVICES=x python demo.py
```

## Train
Single card training:
```bash
cd aagt/experiment
CUDA_VISIBLE_DEVICES=x python trainval.py
```
If you want to adopt multi-GPU training, you can use the following command:
```bash
CUDA_VISIBLE_DEVICES=GPUs python trainval.py
```

## Test
We provide a pre-trained model for testing purposes:
```bash
cd aagt/experiment
CUDA_VISIBLE_DEVICES=1 python test.py --snapshot /../../pretrained.pth.tar
```

## Dataset Description
The training dataset consists of 425 orthodontic patients, yielding a total of 850 paired CBCT and intraoral scan samples. The cohort included 41.9% male subjects, with a mean age of 18.5 years (range: 10–40 years).

CBCT images were acquired using the DCT PRO Dentofacial CBCT system, while intraoral scans were obtained using the 3Shape TRIOS 5 intraoral scanner. The dataset covers a diverse range of clinical conditions, including normal occlusion (72.1%), severe dental crowding (26.0%), missing teeth (2.4%), and subjects undergoing fixed orthodontic treatment with brackets (0.5%).

All data were de-identified prior to use and were collected from Peking University School and Hospital of Stomatology. The study was approved by the institutional ethics committee (Approval No. PKUSSIRB-2025115199). Dataset can be available at https://pan.quark.cn/s/fd2dc541f9ef now.
