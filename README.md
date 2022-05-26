# CGP-DD
Use Cartesian Genetic Programming(CGP) to search effective neural network architecture for image restoration task, typically for unlearned neural network (etc. Deep image prior, Deep decoder)

Code adpated from https://github.com/sg-nm/cgp-cnn-PyTorch[1]

---
Version 0 (Updated at May 26th.)

* Network modules in Deep decoder(DD) [2] are used as the basic function set for CGP, along with Sum and Concat operator;
* Evolution process to search network architecture is provided, retrain process (including saving necessary results in training and testing) is remained to be done;
* Appropriate dataset for training process during network architecture search is required, clean pictures should be put in folder 'img'. 
