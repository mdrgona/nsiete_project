# Neural networks project
**Authors: Marek Drgona, Daniel Pekarcik**

Our goal is to implement a neural network, which is able to recommend jokes to users according to their previous interactions.

# Getting started

Execute the following steps to start training and evaluating model.

## Training via Google Cloud 
1. Create and configure google cloud account and virtual machine (VM) (follow this tutorial - https://github.com/matus-pikuliak/neural_networks_at_fiit/blob/master/project/gcp.ipynb)
2. Clone this repository in your VM: `git clone https://github.com/mdrgona/nsiete_project.git`. **Be sure you are connected to your VM**
3. Go to your cloned repository: `cd nsiete_project`
4. Build docker image: `sudo docker build -t ns_project docker/`
5. Run docker: `sudo docker run --gpus all -it --name recommender --rm -u $(id -u):$(id -g) -p 8888:8888 -p 6006:6006 -v $(pwd):/project/ ns_project`

Now, your docker should be running correctly. **Dont close current terminal and open another. Connect again to your VM. Execute following steps from the second terminal**

6. Open bash console within running docker: `sudo docker exec -it recommender bash`
7. Change directory (into `src/models`) using `cd` command as before
8. Train model: `python train_MLP.py` , or some other file


## Training locally
1. Install docker according to your system
2. Clone this repository: `git clone https://github.com/mdrgona/nsiete_project.git`
3. Go to your cloned repository: `cd nsiete_project`
4. Create docker image: `docker build -t ns_project .`
5. Run docker (unix based system): `sudo docker run -u $(id -u):$(id -g) -p 8888:8888 -p 6006:6006 -v $(pwd):/labs -it ns_project`
6. Open link in terminal, which should open your source tree.


# Project hierarchy
- **dodcs**: documentation
- **docker**: docker files
- **logs**: logs created during training the neural network. Each log has its own timestamp and directory for train and for validate
- **models**: saved trained models
- **notebooks**: data analysis
- **src**: important source codes
  - **data**: files associated to data processing
    - **load_data.py**: data processing
    - **split_data.py**: split dataset 
  - **models**: files associated to model implementation
    - **model_MLP.py**: implementation of MLP model
    - **train_MLP.py**: training pipeline of MLP model
    - **model_GMF.py**: implementation of GMF model
    - **train_GMF.py**: training pipeline of GMF model
    - **model_both.py**: implementation of MLP + GMF
    - **train_both.py**: training pipeline of MLP + GMF
    - **predict.py**: implementation of evaluate metrics and prediction in general
    - **svd.py**: SVD state-of-the-art model
    - **recommender_model.py**: evaluation of precision@metric
    - **main.ipynb**: tensorboard and results analysis
