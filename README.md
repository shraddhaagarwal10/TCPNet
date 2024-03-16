# TCPNet by Biomedical Data Science Lab IISER Bhopal

To load the TCPNet model:
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

