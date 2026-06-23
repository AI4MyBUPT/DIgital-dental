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

cd aagt/extentions/pointops/
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
CUDA_VISIBLE_DEVICES=1 python test.py --snapshot /../../pretrained.pth.tar --benchmark 3DS
```
