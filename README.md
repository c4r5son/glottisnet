# glottisnet

This repository holds a neural network to bound the glottis for endotracheal intubation. Install by running the setup.sh script by the command "bash setup.sh".

To use on an image call 
    "python glottisnet.py -i /path/to/image [--model real]

If you want to use the model trained on only real images provide the --model and specify real. If you want the model trained on images from the an intubation manikin leave off the --model flag, or specify dummy.