# TCPNet: A Novel Tumor Contour Prediction Network using MRIs
In this work, we propose a novel deep-learning framework named TCPNet, which is developed in the spirit of the U-Net model. The proposed architecture ensures that the model segments the tumor contours and explicitly estimates data and model uncertainties in the predictions, which is essential for tumor contour detection. Our extensive study on two open-source brain MRI datasets shows that TCPNet performs better than U-Net and other state-of-the-art in terms of common evaluation metrics. Additionally, the proposed model presents uncertainties in model predictions, demonstrating confidence in segmented tissues or advising for expert intervention if necessary.

## Prerequisites
The following libraries have to be installed one by one before running the code, if they are not already installed. 

[NumPy](https://numpy.org/install/), [Python 3.7 or later version](https://www.python.org/downloads/), [Torch](https://pypi.org/project/torch/), [Transformers](https://pypi.org/project/transformers/)

## How to run the framework?

```
import torch
repo = 'shraddhaagarwal10/TCPNet'  
model = torch.hub.load(repo,  
                       'TCPNet',  
                       force_reload=True)  
```
To load Bayesian Categorical Crossentropy Loss:
```
 bcc_loss = torch.hub.load(repo,
                     'bayesian_categorical_crossentropy',
                       force_reload=True,
                        T = 30,
                        num_classes = 2) 
```
## Contact

For any further query, comment or suggestion, you may reach out at tanmay@iiserb.ac.in

## Citation
```
@inproceedings{tcpnet24,
  title={TCPNet: A Novel Tumor Contour Prediction Network using MRIs},
  author={Shraddha Agarwal, Vinod Kumar Kurmi, Abhirup Banerjee, Tanmay Basu},
  booktitle={Proceedings of IEEE International Conference on Healthcare Informatics},
  pages={},
  year={2014}
}
