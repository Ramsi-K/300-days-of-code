# <center>300 Days Of Code

This repository is a 300-day coding challenge focused on vision technologies.  The repository serves as a comprehensive log of the journey, providing insights into the progress and evolution of skills.
<br>
**[<p align="right">Jump to Daily Task Table](#daily-tasks)**</p>

## <center>Coding Journey

Welcome to my 300-day coding challenge focused on vision technologies! This repository documents my daily coding efforts in the realm of computer vision, encompassing tasks such as semantic segmentation, object detection, classification, reinforcement learning, and GANs. I will also be solving DSA problems from LeetCode on some days to improve my python skills. The 300 days would also include some general python based projects to showcase and improve my skills. The goal is to actively code for at least 1 hour a day for 300 days in the year 2024.
<br><br>

## <center>Projects Undertaken

|  |Project Title        | Description                            | Framework     |  Comments | |
|---|:----------------------:|:---------------------------------------------:|:---------------:|----------:|----------|
| 1 | Road Sign Classifier | Multiclass classification of road sign images    | Pytorch    |  Building training and tracking pipelines from scratch |🟢|
| 2 | Human Action Recognition | Video based multiclass classification of human actions            | TensorFlow    |  In Progress: training baseline models |🟢|

*🟠 : To Do
🟢 : In Progress
🟣 : Complete*
<br/><br/>

## <center>Latest Update

### Converting the [VQGAN paper](https://arxiv.org/abs/2012.09841) to code from scratch
<!-- ----------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
                CHANGE TITLE AND DATE AND DESCRIPTION
                CHANGE TITLE AND DATE AND DESCRIPTION
                CHANGE TITLE AND DATE AND DESCRIPTION
---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- -->
**<p align="right">2024-02-26</p>**

- **Task Description**:
Today, I dedicated my efforts to coding the [VQGAN (Vector Quantized Variational Autoencoder Generative Adversarial Network) paper](https://arxiv.org/abs/2012.09841) to model from scratch. The task involves meticulously translating the intricate details outlined in the research paper into functional code, including defining the architecture, implementing the training loop, and incorporating techniques like vector quantization and adversarial training. It is a challenging yet rewarding task that will allow me to deepen my understanding of generative modeling and neural network architectures.
<br/><br/>

## <center>Purpose

The primary objectives of this coding journey are:

1. **Showcasing Abilities**: Demonstrate my deep learning and coding skills through daily updates, ranging from small coding challenges to more extensive deep learning projects.

2. **Continuous Learning**: Actively engage in learning and improvement, exploring new concepts, algorithms, and frameworks within the field of computer vision.

3. **Building a Portfolio**: Create a comprehensive portfolio of projects and code snippets that highlight proficiency and growth in vision technologies.
<br/><br/>

## <center>The Challenge

- **DailyLogs**: Daily log and description of task undertaken.

- **Projects**: Subfolders containing individual projects, each focused on a specific aspect of vision technologies.

- **CodingChallenges**: Code snippets or solutions from coding challenges, providing a mix of practical coding skills and problem-solving capabilities.

<br/><br/>

## <center>Daily Tasks

Here's a log of the daily tasks completed during the coding challenge:

<br/><br/>

| Day | Date       | Task Description                                       | Tags|
|----|----------------|:--------------------------------------------------------:|-------|
|24| 2024-02-26| Implementing [VQGAN paper](https://arxiv.org/abs/2012.09841) from scratch| GANs|
|23| 2024-02-24|Trained a [multimodal GAN](https://github.com/Ramsi-K/GANs/blob/main/MultimodalGeneration.ipynb) to generate image from text using pretrained CLIP ('ViT-B/32') and Taming Transformers (VQGAN) pretrained models| GANs|
|22| 2024-02-23|Working on [multimodal GAN](https://github.com/Ramsi-K/GANs/blob/main/) architecture to generate image from text| GANs|
|21| 2024-02-22| Trained a [basic GAN](https://github.com/Ramsi-K/GANs/blob/main/Basic%20GAN.ipynb) on the MNIST datasetand an [advanced GAN](https://github.com/Ramsi-K/GANs/blob/main/Advanced%20GAN.ipynb) architecture on the celebA dataset; WANDB tracking [here](https://wandb.ai/ramsik/wgan?workspace=user-ramsik)| GANs|
|20| 2024-02-20| Finished implementing the [ProGAN](https://github.com/Ramsi-K/paper-to-code/tree/main/ProGAN) paper from Scratch in PyTorch. Currently Training on the CelebA-HQ dataset!| GANs|
|19| 2024-02-19| Implementing the [ProGAN](https://github.com/Ramsi-K/paper-to-code/tree/main/ProGAN) paper from Scratch in PyTorch.| GANs|
|18| 2024-02-18| Implemented the [CycleGAN](https://github.com/Ramsi-K/paper-to-code/tree/main/CycleGAN) paper from Scratch in PyTorch. Trained for 150 epochs on a custom car2damagedcar dataset| GANs|
|17| 2024-02-17| Implemented the [pix2pix](https://github.com/Ramsi-K/paper-to-code/tree/main/pix2pix) paper from Scratch in PyTorch. Training for 500 epochs on the Maps Dataset| GANs|
|16| 2024-02-16| Implemented the [WGAN](https://github.com/Ramsi-K/paper-to-code/tree/main/WGAN) and [WGAN-GP](https://github.com/Ramsi-K/paper-to-code/tree/main/WGAN-GP) papers from scratch in PyTorch and trained them on the MNIST dataset| GANs|
|15| 2024-02-15| Implemented the [DCGAN model from scratch](https://github.com/Ramsi-K/paper-to-code/tree/main/DCGAN) from scratch in PyTorch and trained on the MNIST dataset|<div style="white-space: nowrap;"> GANs|
|14|2024-02-14| Trained a Semantic Segmentation model with [Open3D](https://github.com/isl-org/Open3D) and [Open3D-ML](https://github.com/isl-org/Open3D-ML) packages with PyTorch on [SemanticKITTI](http://www.semantic-kitti.org/) dataset |DL 3D|
|13|2024-02-13| Explored the [Open3D](https://github.com/isl-org/Open3D) and [Open3D-ML](https://github.com/isl-org/Open3D-ML) packages and performed data loading, tranformation and visualization tasks. |DL 3D|
|12| 2024-02-12| Trained a simple 2 layer model to play the classic [Snake](https://github.com/Ramsi-K/reinforcement-learning) game in Pytorch | RL|
|11| 2024-02-10| Trained two models in Pytorch on the ViT architecture for [Multiclass Road Sign Classifier](https://github.com/Ramsi-K/multiclass-classification-pytorch). | DL 2D|
|10| 2024-02-09| Built pipelines for dataset manipulation and training in Pytorch for [Multiclass Road Sign Classifier](https://github.com/Ramsi-K/multiclass-classification-pytorch). | DL 2D|
|9| 2024-02-07 | [Hugging Face RL course](https://github.com/huggingface/deep-rl-class) completed units 7, 8a, 8b and advanced topics. [Certificate](https://github.com/Ramsi-K/reinforcement-learning/blob/main/images/HFDRL-cert.png?raw=true) | RL |
|8| 2024-02-06 | [Hugging Face RL course](https://github.com/huggingface/deep-rl-class) completed units 4, 5 and 6. | RL |
|7 | 2024-02-03| LeetCode problems: [11-container-with-most-water](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/11-container-with-most-water) and [26-remove-duplicates-from-sorted-array](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/26-remove-duplicates-from-sorted-array) | DSA |
|6 | 2024-02-01| Explored datasets, structured project and trained EfficientNet_B0 model for [MultiClass Human Action Classification](https://github.com/Ramsi-K/video-classification-tf) from **video data** | DL 3D |
| 5 | 2024-01-31  | Explored datasets, conducted EDA, and structured project for [Multiclass Road Sign Classifier](https://github.com/Ramsi-K/multiclass-classification-pytorch). | <div style="white-space: nowrap;">DL 2D|
| 4  | 2024-01-29 | [Implementing Vision Transformer](https://github.com/Ramsi-K/paper-to-code/tree/main/ViT) (ViT) model from scratch in PyTorch. | DL 2D|
| 3  | 2024-01-28 |   LeetCode problems:  [1-two-sum](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1-two-sum), [2-add-two-numbers](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/2-add-two-numbers), [4-median-of-two-sorted-arrays](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/4-median-of-two-sorted-arrays) | DSA|
| 2  | 2024-01-27 | Explored [classic control tasks](https://github.com/Ramsi-K/reinforcement-learning/blob/main/Classic_Control_exploration.ipynb); studied MDP, TD, Monte Carlo, [Q-Learning](https://github.com/Ramsi-K/reinforcement-learning/blob/main/Q-learning/README.md) theory | RL|
| 1  | <div style="white-space: nowrap;">2024-01-26 | [MDP basics exploration](https://github.com/Ramsi-K/reinforcement-learning/blob/main/Basics_of_Markov_Decision_Process.ipynb) on custom Maze env with random policy exploration.| RL|

---
<br/><br/>
Feel free to reach out, provide feedback, or collaborate on any aspect of the journey. Let's embark on this coding adventure together!

Happy Coding! 🚀

<!-- TODO Topics, Ideas, Tutorials
**Topics to cover:** Segmentation, Object Detection, human pose, SLAM, GANs, Quantization, Pruning, Depth Analysis, Multi-Object Multi-Camera Tracking, 3D reconstruction, augmented reality?, Image Restoration, Image Enhancement, Optical Flow, Multi-View Geometry, Domain Adaptation, Anomaly Detection, 3D point clouds, OpenCV implementations

**Project Ideas:** 
- Hand gesture (ASL?), 
- Vehicle Tracking on road, 
- Point Cloud Library, 
- [Multi-object tracking of people in surveillance videos](https://youtu.be/1VTQ2b3fbb0)
- [A  prototype of a virtual fitting room solution for a startup from Silicon Valley](https://youtu.be/HloNNIW1kx4)
- An engine for visual understanding of industrial scenes captured by high-precision laser scanners (2D and 3D data)
- Aerial and satellite image analysis – semantic segmentation of multispectral photos and objects detection

**Tutorials:** 
- PCL 
    - Hypothesis Verification for 3D Object Recognition 
    -  Detecting people on a ground plane with RGB-D data
    - Detecting people and their poses using PointCloud Library -->
