# Neural networks project
**Authors: Marek Drgona, Daniel Pekarcik**

Our goal is to implement a neural network, which is able to recommend jokes to users according to their previous interactions.

# How to run project
TODO

# Project hierarchy
- **docker**: docker files
- **logs**: logs created during training the neural network. Each log has its own timestamp and directory for train and for validate
- **notebooks**: jupyter notebooks, consists of data analysis and our experiments with implementation
- **src**: important source codes
  - **data**: files associated to data processing
  - **models**: files associated to model implementation
    - **model.py**: implementation of joke recommender
    - **train.py**: training pipeline
    - **predict.py**: implementation of evaluate metrics and prediction in general
    - **main.ipynb**: main file, where we are able to run training
- **Evaluator.ipynb**: state of the art recommender implementation (not deep learning)
- **Project_proposal.ipynb**: proposal of project including motivation, related work and described dataset
