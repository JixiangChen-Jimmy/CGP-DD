# CGP-DD
Use Cartesian Genetic Programming(CGP) to search effective neural network architecture for image restoration task, typically for unlearned neural network (e.g. Deep image prior, Deep decoder)

Code adpated from https://github.com/sg-nm/cgp-cnn-PyTorch [1]

---
Version 0 (Updated at May 26th.)

* Network modules in Deep decoder(DD) [2] are used as the basic function set for CGP, along with Sum and Concat operator;
* Evolution process to search network architecture is provided, retrain process (including saving necessary results in training and testing) is remained to be done;
* Appropriate dataset for training process during network architecture search is required, clean pictures should be put in folder 'img'. 
---
Reference:

1. Masanori Suganuma, Shinichi Shirakawa, and Tomoharu Nagao, "A Genetic Programming Approach to Designing Convolutional Neural Network Architectures," Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17, Best paper award), pp. 497-504 (2017)
2. Heckel, Reinhard & Hand, Paul. "Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks, " (2018) 
