conda create -y -n shoebox python=3.8 pip
conda activate shoebox
conda install -y pytorch=1.7 torchvision cudatoolkit=10.1 -c pytorch
conda install -y pandas scikit-image scikit-learn 
conda install -y -c conda-forge gdcm

pip install albumentations kaggle iterative-stratification omegaconf pretrainedmodels pydicom timm transformers
