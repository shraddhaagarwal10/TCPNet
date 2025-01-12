# TCPNet: A Novel Tumor Contour Prediction Network using MRIs
In this work, we propose a novel deep-learning framework named [TCPNet](https://ieeexplore.ieee.org/document/10628585), which is developed in the spirit of the U-Net model. The proposed architecture ensures that the model segments the tumor contours and explicitly estimates data and model uncertainties in the predictions, which is essential for tumor contour detection. Our extensive study on two open-source brain MRI datasets shows that TCPNet performs better than U-Net and other state-of-the-art in terms of common evaluation metrics. Additionally, the proposed model presents uncertainties in model predictions, demonstrating confidence in segmented tissues or advising for expert intervention if necessary.

## Prerequisites
The following libraries have to be installed one by one before running the code, if they are not already installed. 

[NumPy](https://numpy.org/install/), [Python 3.7 or later version](https://www.python.org/downloads/), [Torch](https://pypi.org/project/torch/), [Transformers](https://pypi.org/project/transformers/)

## Clone the repository
```
git clone https://github.com/shraddhaagarwal10/TCPNet-A-Novel-Tumor-Contour-Prediction-Network-using-MRIs.git
```

## How to download the Brain tumor dataset?
The [Brain tumor dataset](https://figshare.com/articles/dataset/brain-tumor-dataset/1512427) has been used.

In the **Dataset** folder, run <ins>download\_data.sh</ins> shell script to download all data. It contains 3064 MRI images and 3064 masks. Run the following command to download the data.
```
cd Dataset
sh download_data.sh
```
After that run the following command to convert the data into numpy format.
```
python3 mat_to_numpy.py brain_tumor_dataset
```
The numpy files have been saved in **brain_tumor_dataset** folder.

## How to run the framework?

### For TCPNet with only model uncertainty:
```
python3 main_model_unc.py --data-path='Dataset/train_tumor_dataset/' --epochs=200 --batch-size=32 --lr=1e-3 --sample=20 --classes=2
```

### For TCPNet with both data and model uncertainty:
```
python3 main_data_model_unc.py --data-path='Dataset/brain_tumor_dataset/' --epochs=200 --batch-size=32 --lr=1e-3 --sample=20 --lamda=0.01 --classes=2
```

## How to use TCPNet model?
```
import torch
repo = 'shraddhaagarwal10/TCPNet'  
model = torch.hub.load(repo,  
                       'TCPNet',  
                       force_reload=True)  
```
## How to use Bayesian Categorical Crossentropy Loss?
```
 bcc_loss = torch.hub.load(repo,
                     'bayesian_categorical_crossentropy',
                       force_reload=True,
                        T = 30,
                        num_classes = 2) 
```
## Contact

For any further query, comment or suggestion, you may reach out at tanmay@iiserb.ac.in and shraddhaagarwal2001@gmail.com

## Citation
```
@inproceedings{tcpnet24,
  title={TCPNet: A Novel Tumor Contour Prediction Network using MRIs},
  author={Shraddha Agarwal, Vinod Kumar Kurmi, Abhirup Banerjee, Tanmay Basu},
  booktitle={Proceedings of IEEE International Conference on Healthcare Informatics},
  pages={},
  year={2024}
}
