# FacialAtrributeAI
COMP3419 Assignment, making a facial atrribute analyzer AI based on previous research articles using CelebA dataset, preweighted training via ImageNet and EfficientNet as backbone model. 

Inorder to run the model, do `python3 model_final.py` while being in the `src` directory. Make sure to install all needed libraries via pip, then training should begin as normal where a loading bar should show metrics of EST of training per epoch

Inorder to evaluate the model, simply run `python3 eval.py`

Inorder to try the model after training use the `unified_runner.py`. This contains the source code to run the older models also. For the final model just type `python3 unified_runner.py final ./images/image_1.png`. A stock image is included.

some notable issues could arrise during installation of pytorch, because of this please try `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` if other installation methods fail.(Further mentioned in `requirements.txt` which can be run directly as well)