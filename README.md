# <center>👨‍💻✨🔭300 Days Of Code 🤖🎉🚀

 This repository is a 300-day coding challenge focused on vision technologies. The repository serves as a comprehensive log of the journey, providing insights into the progress and evolution of skills.
Get ready for 300 days of coding excitement, challenges, and triumphs in the universe of computer vision!

**[Jump to Daily Task Table](#daily-tasks)**

## <center>Coding Journey

Welcome to my 300-day coding challenge focused on vision technologies! This repository documents my daily coding efforts in the realm of computer vision, encompassing tasks such as semantic segmentation, object detection, classification, reinforcement learning, and GANs. I will also be solving DSA problems from LeetCode on some days to improve my python skills. The 300 days would also include some general python based projects to showcase and improve my skills. The goal is to actively code for at least 1 hour a day for 300 days in the year 2024.
<br><br>

## <center>Projects Undertaken

|  |Project Title        | Description                            | Framework     |  Comments | |
|---|:----------------------:|:---------------------------------------------:|:---------------:|----------:|----------|
| 1 | LunaNet3D | 3D medical image analysis for lung nodule detection using the LUNA16 dataset | PyTorch | Working on data preprocessing, transformations, and visualizations |🟢|
| 2 | Road Sign Classifier | Multiclass classification of road sign images | PyTorch | Building training and tracking pipelines from scratch |🟠|
| 3 | Human Action Recognition | Video-based multiclass classification of human actions | TensorFlow | In Progress: training baseline models |🟠|

*🟠 : On Pause
🟢 : In Progress
🟣 : Complete*
<br/><br/>

## <center>Latest Update
<!-- ----------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
                CHANGE TITLE AND DATE AND DESCRIPTION
                CHANGE TITLE AND DATE AND DESCRIPTION
                CHANGE TITLE AND DATE AND DESCRIPTION
---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- -->

### [Notebookify](https://github.com/Ramsi-K/notebookify)

**<p align="right">2024-07-03</p>**

- **Task Description**:  

I expanded the script to include Google Drive integration so I could share converted notebooks more easily. I spent most of today debugging path inconsistencies during uploads. By the end of the day, the script was functional, but it still didn't handle diverse output types.

---

## <center>The Challenge

Embark on a thrilling 300-day coding odyssey, a quest where every day is a new adventure in the realm of computer vision and deep learning. Join me on this exciting journey of practical coding tasks, where each day unfolds with hands-on challenges, research paper implementations, and real-world problem-solving.

**Here's what makes this challenge an epic adventure:**

- **Hands-on Coding**: Dive deep into practical coding tasks, from implementing cutting-edge research papers to tackling real-world problems head-on.

- **Continuous Learning**: Embrace a culture of lifelong learning, exploring new concepts, algorithms, and frameworks in the dynamic field of vision technologies.

- **Beyond Boundaries:** Explore the frontiers of computer vision and deep learning, pushing the limits with projects that go from semantic segmentation to GANs, reinforcement learning, and more.

- **Building a Robust Portfolio**: Craft a comprehensive portfolio of projects and code snippets, showcasing not just skills, but the journey of growth and innovation.

- **Progressive Learning:** Witness the evolution of skills as each day adds new layers of expertise, building a solid foundation and demonstrating continuous improvement.

- **Meaningful Contributions**: Connect, collaborate, and share insights with a growing community of enthusiasts, making this journey a collective exploration of the fascinating world of vision technologies.
<br/><br/>

## <center>Challenge Structure

- **DailyLogs**: Daily log and description of task undertaken.

- **Projects**: Repositories and subfolders containing individual projects, each focused on a specific aspect of vision technologies.

- **CodingChallenges**: Code snippets or solutions from coding challenges, providing a mix of practical coding skills and problem-solving capabilities.
<br/><br/>

## 30-Day Coding Sprints: Project Highlights

### Latest Sprint: Days 91-120 Highlights

1. **YOLOv3 Implementation**:  
   - Focused on debugging and refining YOLOv3 components, including the loss function, target assignment, and bounding box handling.  
   - Resolved dataset generation issues and verified mask populations, ensuring accurate object detection assignments across scales.  
   - Successfully aligned image dimensions with grid sizes and tested loss calculations component-wise, progressing toward a robust YOLOv3 implementation.

2. **LunaNet3D Data Pipeline Enhancements**:  
   - Implemented a balanced data loader with dynamic sampling, adaptive augmentation, and difficulty-based sampling.  
   - Reduced training times significantly by optimizing GPU utilization and caching strategies, achieving 20 minutes per epoch for the minimal model.  
   - Addressed issues in batching, indexing, and augmentation, culminating in the successful training of baseline and minimal models with custom data loaders.

3. **MLOps with ZoomCamp**:  
   - Completed Week 3 assignments on orchestration, covering pipeline development, experiment tracking, and model management.  
   - Overcame Docker setup challenges and fine-tuned workflows for effective orchestration.  
   - Implemented MLflow logging for model and artifact tracking in both local and containerized environments.

4. **Paper-to-Code Repository Organization**:  
   - Refactored and organized the repository with a master README, individual project documentation, requirements files, and environment setups.  
   - Enhanced clarity and modularity, making the repository easier to navigate and extend for future projects.  

5. **LeetCode Practice**:  
   - Completed a series of SQL-based problems, strengthening query optimization and logical reasoning skills.  
   - Solved problems on topics like employee bonuses, customer orders, and data aggregation.

### Key Themes:

- **Efficient Pipelines**: Streamlined LunaNet3D and YOLOv3 workflows with improved data loaders, augmentation strategies, and debugging.  
- **Scalable MLOps Practices**: Applied foundational concepts of orchestration and experiment tracking to real-world setups.  
- **Repository Management**: Enhanced organization and documentation for long-term usability and professional presentation.  
- **Consistent Skill Development**: Balanced technical learning with coding problem-solving to maintain diverse skill sets.


---

<details>
  <summary>Archived 30-Day Sprints</summary>

### Archived Sprint: Days 61-90 Highlights

1. **Luna-Net3D-Archived Data Validation and Visualization**: An intensive focus on ensuring data quality and alignment, developing scripts for voxel-to-lung validation, boundary checks, and comparative plotting of aligned and misaligned nodules in both 2D and 3D. This phase was crucial for cleaning up the dataset and ensuring accurate annotations.

2. **YOLOv3 Paper Implementation**: Started implementing YOLOv3 from scratch based on the original paper to understand the architecture of YOLO and its layers. YOLO’s object detection architecture is an excellent candidate for applying nodule detection to the Luna-Net3D-Archived dataset. The work involved coding the layers, training on preliminary data, and drafting detailed notes on implementation.

3. **Exploring MLOps with ZoomCamp**: Completed Weeks 1 and 2 of the MLOps ZoomCamp course, covering foundational MLOps concepts, experiment tracking, and model management with MLflow. Weekly modules included hands-on exercises, implementing experiment tracking, and setting up a model registry to organize experiments and streamline model development.

4. **Data Preprocessing and Augmentation for Luna-Net3D-Archived**: Developed effective data transformations and resizing methods to improve data loading and training speeds. This involved extensive exploration of TorchIO for 3D data augmentation, implementing padding, resizing, and balancing methods, and tackling augmentation-related debugging challenges.

### Archived Sprint 2: Days 31-60 Highlights
  
  1. **LunaNet3D - Medical Image Preprocessing**: Delved into data preprocessing, manipulation, and visualization of the LUNA16 dataset. This sprint involved detailed exploration of medical images, working on CT scan fundamentals, and generating insightful 3D visualizations to better understand the dataset. I tackled tasks like thresholding, segmentations, and transformations.

  2. **Graph Neural Networks (GNNs) with PyG**: Completed various tasks with GNNs using PyTorch Geometric (PyG), such as node and graph classification, understanding spectral graph convolutions, and working on point cloud classification using PointNet++. Key projects included GAT models and link prediction on the MovieLens dataset.

  3. **LeetCode DSA Practice**: Strengthened problem-solving skills by solving LeetCode problems on topics such as permutations, binary search, and array manipulations. Continued sharpening algorithmic thinking with hands-on exercises in preparation for coding interviews.

  4. **3D Object Detection**: Explored 3D object detection by implementing models like Frustum PointNets and VoteNet. These models are key for real-time object detection in 3D environments, using point cloud data and voxel representations to enhance object recognition capabilities.

### Archived Sprint 1: Days 1-30 Highlights
  
  1. **Implementing Vision Transformer (ViT) from Scratch**: Developing a deep understanding of the ViT architecture and translating theoretical concepts into functional code to create a ViT model using PyTorch.

  2. **Training a Semantic Segmentation Model with Open3D**: Leveraging the Open3D library to train a semantic segmentation model on the SemanticKITTI dataset, involving data loading, transformation, and visualization tasks.

  3. **Exploring Classic Control Tasks for Reinforcement Learning**: Delving into classic control environments to understand Markov Decision Processes (MDP), Temporal Difference (TD) learning, and Q-learning, implementing these concepts in Python using reinforcement learning techniques.

  4. **Building a Multimodal GAN for Image Generation**: Constructing a Generative Adversarial Network (GAN) capable of generating images from text descriptions by combining pre-trained models such as CLIP and VQGAN, emphasizing multi-modal fusion and learning.

</details>

<br/><br/> 

## <center>Daily Tasks

Here's a log of the daily tasks completed during the coding challenge:

| Day | Date       | Task Description                                       | Tags|
|----|----------------|:--------------------------------------------------------:|-------|
|141| 2024-07-03 | [Notebookify](https://github.com/Ramsi-K/notebookify):  Debugged path inconsistencies during uploads. | DevTools |
|140| 2024-07-02 | [Notebookify](https://github.com/Ramsi-K/notebookify):  Began drafting `convert_to_markdown.py` as a standalone script. | DevTools |
|139| 2024-07-01 | [Notebookify](https://github.com/Ramsi-K/notebookify): Debugged issues combining nbconvert with Google Drive uploads. | DevTools |
|138| 2024-06-30 | LeetCode: [DailyLeadsandPartners](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1693-daily-leads-and-partners/); [NumberofEmployees](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1731-the-number-of-employees-which-report-to-each-employee/); [TotalTimeSpent](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1741-find-total-time-spent-by-each-employee/);  [RecyclableLowFatProducts](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1757-recyclable-and-low-fat-products/); [PrimaryDepartment](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1789-primary-department-for-each-employee/); [RearrangeProductsTable](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1795-rearrange-products-table/) | DSA |
|137| 2024-06-29 | [Notebookify](https://github.com/Ramsi-K/notebookify): Began integrating Google Drive API for file sharing. | DevTools |
|136| 2024-06-28 | [Notebookify](https://github.com/Ramsi-K/notebookify): Explored Git LFS for large notebooks; deemed unsuitable. | DevTools |
|135| 2024-06-27 | [Notebookify](https://github.com/Ramsi-K/notebookify): Extended testing `nbconvert` for Markdown conversion; faced rendering issues. | DevTools |
|134| 2024-06-26 | [Notebookify](https://github.com/Ramsi-K/notebookify):  Researched GitHub Markdown limitations after the last notebook would not render on Github and also resulted in 85% capacity for GitLFS; explored `nbconvert` as a solution. | DevTools |
|133| 2024-06-25 | [Monocular Depth Estimation](https://github.com/Ramsi-K/3D-Vision-Playground/Depth-Estimation/Monocular): Monocular depth estimation [self-implementation](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/Depth-Estimation/Monocular/02_SelfExploration_MonocularDepth_DPT.ipynb) using pretrained HF DPT; RMSE and SSIM evaluated; point clouds using Open3D | 3D CV |
|132| 2024-06-24 | [Monocular Depth Estimation](https://github.com/Ramsi-K/3D-Vision-Playground/Depth-Estimation/Monocular): Studied monocular depth estimation and worked through FiftyOne [Tutorial](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/Depth-Estimation/Monocular/01_Basics_MonocularDepth_Tutorial.ipynb) for Sun-RGBD dataset using DPT and Marigold | 3D CV |
|131| 2024-06-23 | LeetCode: [PatientsWithCondition](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1527-patients-with-a-condition/); [CustomersWhoVisited](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1581-customer-who-visited-but-did-not-make-any-transactions/); [BankAccountSummary](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1587-bank-account-summary-ii/); [PercentageUsersAttendedConcert](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1633-percentage-of-users-attended-a-contest/); [FixNamedInTable](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1667-fix-names-in-a-table/); [InvalidTweets](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1683-invalid-tweets/)  | DSA |
|130| 2024-06-22 | LeetCode: [StudentsandExaminations](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1280-students-and-examinations/); [ListProductsOrdered](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1327-list-the-products-ordered-in-a-period/); [ReplaceEmployeeId](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1378-replace-employee-id-with-the-unique-identifier/); [TopTravellers](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1407-top-travellers/); [GroupSoldProducts](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1484-group-sold-products-by-the-date/); [FindUsersWithEmails](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1517-find-users-with-valid-e-mails/) | DSA |
|129| 2024-06-21 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived) Resolved difficulty score application and prefetch wrapper issues and optimized training verbosity | 3D CV |
|128| 2024-06-20 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived) Changed training metrics from loss and accuracy to account for precision, recall, F1 metrics; integrated balanced F1 for early stopping; updated train and validate methods | 3D CV |
|127| 2024-06-19 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived) Adjusted the baseline model to improve its integration of spatial and coordinate features. Resolved multiple shape mismatch issues that caused training crashes. | 3D CV |
| 126 | 2024-06-18 | [MLOps ZoomCamp](https://github.com/Ramsi-K/mlops-zoomcamp) Week 4 : Deployment - Implemented the homework assignment: trained a model, deployed it as a REST API using Flask and Docker, and tested deployment scenarios. Experimented with MLflow model registry integration for deployments. | MLOps |
| 125 | 2024-06-17 | [MLOps ZoomCamp](https://github.com/Ramsi-K/mlops-zoomcamp) Week 4 : Deployment - focusing on deploying models as web services, batch models, and streaming services | MLOps |
| 124 | 2024-06-16 | LeetCode: [ProjectEmployees](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1075-project-employees-i/); [SalesAnalysis](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1084-sales-analysis-iii/); [UserActivity](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1141-user-activity-for-the-past-30-days-i/); [ArticleViews](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1148-article-views-i/); [MarketAnalysis](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1158-market-analysis-i/); [ProductPrice](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1164-product-price-at-a-given-date/); [ReformatDepartment](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1179-reformat-department-table/) | DSA |
| 123 | 2024-06-15 | LeetCode: [ExchangeSeats](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0626-exchange-seats/); [SwapSalary](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0627-swap-salary/); [CustomersWhoBoughtAllProducts](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1045-customers-who-bought-all-products/); [ActorsandDirectors](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1050-actors-and-directors-who-cooperated-at-least-three-times/); [ProductSalesAnalysis1](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1068-product-sales-analysis-i/); [ProductSalesAnalysis3](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/1070-product-sales-analysis-iii)| DSA |
| 122 | 2024-06-14 | LeetCode: [FriendRequests](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0602-friend-requests-ii-who-has-the-most-friends); [SalesPerson](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0607-sales-person/); [TreeNode](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0608-tree-node/); [TriangleJudgement](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0610-triangle-judgement/); [BiggestSingleNumber](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0619-biggest-single-number/); [NotBoringMovies](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0620-not-boring-movies)| DSA |
| 121 | 2024-06-13 | [YOLOv3](https://github.com/Ramsi-K/paper-to-code/tree/main/YOLOv3):  Debugging shape errors; extensive debugging on dataset processing and anchor assignments configurations| 2D CV |
| 120 | 2024-06-12 | [YOLOv3](https://github.com/Ramsi-K/paper-to-code/tree/main/YOLOv3):  Resolved dataset target generation issues and adjusting bounding boxes accordingly. Validated obj_mask and noobj_mask populations, ensuring object assignments were non-zero across all scales. Tested and confirmed alignment of image dimensions and target grid sizes | 2D CV |
| 119 | 2024-06-11 | [YOLOv3](https://github.com/Ramsi-K/paper-to-code/tree/main/YOLOv3):  Debugged the YOLOLoss function and verified component-wise loss breakdown with dummy targets; Started refining target assignment for obj_mask and noobj_mask in dataset.py, verified initial target setups for all scales | 2D CV |
| 118 | 2024-06-10 | [YOLOv3](https://github.com/Ramsi-K/paper-to-code/tree/main/YOLOv3):  Improved anchor dimension handling in YOLOLoss. Successfully calculated loss_x, loss_y, loss_w, and loss_h individually for each anchor index. Encountered NaN in total loss, to be investigated tomorrow; [MLOps ZoomCamp](https://github.com/Ramsi-K/mlops-zoomcamp) Week 3 Orchestration Homework  | 2D CV, MLOps |
| 117 | 2024-06-09 | [YOLOv3](https://github.com/Ramsi-K/paper-to-code/tree/main/YOLOv3): Revised and Refactored YOLOv3 architecture and dataset functions; resolved model input mismatches, fixed dataset logic, and refined loss calculations; [MLOps ZoomCamp](https://github.com/Ramsi-K/mlops-zoomcamp) Week 3 Orchestration  | 2D CV, MLOps |
| 116 | 2024-06-08 | [Paper to Code](https://github.com/Ramsi-K/paper-to-code): Organized repository with master README, requirements files, and environment setup | Documentation |
|115| 2024-06-07 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived)Completed balanced data loader, adaptive augmentation, difficulty-based sampling for batch construction.Minimal and Baseline models successfully training with custom Dataloaders. | 3D CV |
|114| 2024-06-06 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived):Added dynamic sampling and difficulty-based DataLoader reinitialization, logging adjustments, and batch structure validation in training loop. Training on minimal model, sample dataset for debugging.| 3D CV |
|113| 2024-06-05 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived):Implemented dynamic tracking of difficulty and confidence scores in training loop; adjusted PrefetchLoader for accessibility of batch size.(Training time down to 20 min/epoch on minimal)| 3D CV |
|112| 2024-06-04 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived):Resolved batching issues, confirmed balanced sampling, and finalized basic data loader with augmentation | 3D CV |
|111| 2024-06-03 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived):Balanced data loader logic revision, implementation, debugging batch processing, and resolving indexing errors. | 3D CV |
|110| 2024-06-02 | [MLOps ZoomCamp](https://github.com/Ramsi-K/mlops-zoomcamp) Week 3 : Major Docker Issues | MLOps |
|109| 2024-06-01 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived):Balanced data loader modifications, debugging and batch processing debugging , handling indexing issues. | 3D CV |
|108| 2024-05-31 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Balanced data loader creation, model testing optimizations, README update | 3D CV |
|107| 2024-05-30 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): GPU Optimization: bottleneck debugging, minimal model creation, caching strategies. Training time down to 4hrs/epoch on baseline. | 3D CV |
|106| 2024-05-29 | [MLOps ZoomCamp](https://github.com/Ramsi-K/mlops-zoomcamp) Week 2: MLOps experiment tracking with homework | MLOps |
|105| 2024-05-28 | [MLOps ZoomCamp](https://github.com/Ramsi-K/mlops-zoomcamp) Week 2: Experiment tracking and model management with MLflow | MLOps |
|104| 2024-05-27 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Debugging: seed setting, prefetch tuning, worker optimization, mixed precision training | 3D CV |
|103| 2024-05-26 | LeetCode: [EmployeeBonus](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0577-employee-bonus), [FindCustomerReferee](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0584-find-customer-referee), [Investmentsin2016](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0585-investments-in-2016) [CustomerPlacingTheLargestNumberofOrders](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0586-customer-placing-the-largest-number-of-orders/),[BiggestCountries](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0595-big-countries/),[ClassesMoreThan5Students](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0596-classes-more-than-5-students/), [HumanTrafficofStadium](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0601-human-traffic-of-stadium/) | DSA |
|102| 2024-05-25 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Training, metrics, and evaluation scripts with extensive debugging (initial epoch time >160 hours) | 3D CV |
|101| 2024-05-24 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Baseline model for classification completed with theoretical backing | 3D CV |
|100| 2024-05-23 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Classification training scripts and baseline model setup | 3D CV |
|99 | 2024-05-22 |[MLOps ZoomCamp](https://github.com/Ramsi-K/mlops-zoomcamp) Week 1 Homework | MLOps |
|98 | 2024-05-21 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Modularization and system setup, overcoming frustrating system issues | 3D CV |
|97 | 2024-05-20 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Dataloaders with TorchIO for augmentations, tackling major image sizing and augmentation logging issues | 3D CV |
|96 | 2024-05-19 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Voxel recalculation post-resizing, image conversion to .npy format, and dataset organization | 3D CV |
|95 | 2024-05-18 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Data resampling, resizing, padding, and correlation studies on z-range and xy-range | 3D CV |
|94 | 2024-05-17 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Cross-validation setup with official 10-fold structure and train/val/test splits | 3D CV |
|93 | 2024-05-16 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Fixing NaN label values, addressing lung instance splitting across folders | 3D CV |
|92 | 2024-05-15 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Data organization with optimal resizing and storage methods for efficient loading | 3D CV |
|91 | 2024-05-14 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Image transformations and augmentations completed | 3D CV |
|90 | 2024-05-13 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Data preprocessing with resizing, padding, and augmentation exploration | 3D CV |
|89 | 2024-05-12| [YOLOv3](https://github.com/Ramsi-K/paper-to-code/tree/main/YOLOv3) paper implementation, and README documentation. Debugging training. | DL 2D |
|88 | 2024-05-11 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Interactive 3D plotting of misaligned nodules using Plotly | 3D CV |
|87 | 2024-05-10 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Validating misaligned images with 2D/3D plotting and working on interactive 3D visualizations | 3D CV |
|86 | 2024-05-09 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Data validation markdown creation, recalculating voxel distances for annotations and labels | 3D CV |
|85 | 2024-05-08 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Fixed LPI nodule orientation issue with mhd transform matrix adjustments | 3D CV |
|84 | 2024-05-07 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Completed visual inspection, discovered errors in LPI nodule conversion | 3D CV |
|83 | 2024-05-06 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Multiple linear regression analysis, voxel alignment, orientation validation | 3D CV |
|82 | 2024-05-05 | [YOLOv3](https://github.com/Ramsi-K/paper-to-code/tree/main/YOLOv3) paper exploration and code setup | DL 2D |
|81 | 2024-05-04 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Boundary-nodule correlations, p-values, and outlier detection | 3D CV |
|80 | 2024-05-03 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Debugging boundary/edge cases in misaligned nodule analysis | 3D CV |
|79 | 2024-05-02 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Misaligned nodule analysis and boundary detection | 3D CV |
|78 | 2024-05-01 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Voxel to lung alignment validation | 3D CV |
|77 | 2024-04-30 |[LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Data restructuring and comparative visual analysis | 3D CV |
|76 | 2024-04-29 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Data validation script and comparative plotting | 3D CV |
|75| 2024-04-28 | LeetCode: [Game Play Analysis 1](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0511-game-play-analysis-i), [Game Play Analysis 4](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0550-game-play-analysis-iv), [Managers with 5 direct reports](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0570-managers-with-at-least-5-direct-reports) | DSA |
|74| 2024-04-27 |  [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Final data validation and checking for consistency | 3D CV |
|73| 2024-04-26 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Redid voxel transformations, orientation issues, and validation | 3D CV |
|72| 2024-04-25 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Pre-processing voxel transformations and storing them in CSV format | 3D CV |
|71| 2024-04-24 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Helper function files and more data analysis | 3D CV |
|70| 2024-04-23 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Data exploration, outlier handling, class imbalance analysis, and sampler setup | 3D CV |
|69| 2024-04-22 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Addressing annotation mismatches and CSV file investigation | 3D CV |
|68| 2024-04-21 | LeetCode: [Nth Highest Salary](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0177-nth-highest-salary), [Department Highest Salary](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0184-department-highest-salary), [Top 3 Salaries](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0185-department-top-three-salaries), [Delete Duplicate Emails](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0196-delete-duplicate-emails), [Rising Temperature](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0197-rising-temperature), [Trips and Users](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0262-trips-and-users) | DSA |
|67| 2024-04-20 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Statistical analysis, volume comparison setup, utils.py updates | 3D CV |
|66| 2024-04-19 | Revisiting [PAg-NeRF](https://arxiv.org/abs/2309.05339) paper for Gaussian Splatting and potential integration | NeRF |
|65| 2024-04-18 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Visualizations (nodules, binary masking, segmentations) | 3D CV |
|64| 2024-04-17 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Visualizations using thresholding and morphological techniques | 3D CV |
|63| 2024-04-16 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Exploring excluded annotations and understanding the dataset | 3D CV |
|62| 2024-04-15 | Radiance Field Meetup: NeRF and Gaussian Splatting discussion | NeRF |
|61| 2024-04-14 | LeetCode: [Combine two tables](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0175-combine-two-tables), [Second highest salary](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0176-second-highest-salary), [Rank Scores](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0178-rank-scores), [Consecutive Numbers](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0180-consecutive-numbers), [Employees earning more than their managers](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0181-employees-earning-more-than-their-managers), [Duplicate Emails](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0182-duplicate-emails), [Customers who never order](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0183-customers-who-never-order) | DSA |
|60| 2024-04-13 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Data Preprocessing| 3D CV |
|59| 2024-04-12 | [LunaNet3D](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Data Preprocessing -  coordinates manipulations and image transformations | 3D CV |
|58| 2024-04-11 | [Luna16](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Data Preprocessing - Medical image and ct scan fundamanetals and lib explorations| 3D CV |
|57| 2024-04-10 | [Luna16](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Plotting mhd images in 3D and generating animations to better understand data | 3D CV  |
|56| 2024-04-09 | [Luna16](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Plotting mhd images in 2D from sample subset | 3D CV |
|55| 2024-04-08 | [Luna16](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived): Processing mhd images and working on sample dataset | 3D CV |
|54| 2024-04-07 | LeetCode: [Next Permutation](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0031-next-permutation), [Length of Last Word](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0058-length-of-last-word),[Merge Sorted Array](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0088-merge-sorted-array) | DSA |
|53| 2024-04-06 | LeetCode: [Find the Index of the First Occurrence in a String](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0028-find-the-index-of-the-first-occurrence-in-a-string), [Divide Two Integers](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0029-divide-two-integers), [3Sum](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0015-3sum), [4Sum](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0018-4sum),  [Search Insert Position](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0035-search-insert-position) | DSA |
|52| 2024-04-05 | Exploring the [Luna16](https://luna16.grand-challenge.org/) dataset for lung node analysis inlcuding [data exploration](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Luna-Net3D-Archived) on the sample. | 3D CV |
|51| 2024-04-04 | Exploring Graph Neural Networks using PyG: Built and Implemented a [GAT model](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/Scaling_Graph_Neural_Networks.ipynb) on the Cora dataset | GNN |
|50| 2024-04-03 | LeetCode: [Longest Palindromic Substring](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0005-longest-palindromic-substring), [Zigzag Conversion](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0006-zigzag-conversion), [Reverse Integer](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0007-reverse-integer) & [Remove Element](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0027-remove-element) | DSA |
|49| 2024-04-02 | Exploring Graph Neural Networks using PyG: [Link Prediction](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/Link_Prediction_on_MovieLens.ipynb) & [Link Regression](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/Link_Regression_on_Movielens.ipynb) on toy [MovieLens dataset](https://grouplens.org/datasets/movielens/) | GNN |
|48| 2024-04-01 | Exploring Graph Neural Networks using PyG:  Understanding [message passing](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/message_passing.ipynb) and utilization of various [aggregation functions](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/7_Aggregation_Package.ipynb) | GNN |
|47| 2024-03-26 | Exploring Graph Neural Networks using PyG: [Understanding GNN predictions with the Captum lib](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/6_GNN_Explanation.ipynb) and went through a [GNN overview](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/GNN_overview.ipynb) | GNN |
|46| 2024-03-25 | Exploring Graph Neural Networks using PyG: [Point Cloud Classification using PointNet++](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/5_Point_Cloud_Classification.ipynb) using the [GeometricShapes](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.GeometricShapes) dataset | GNN |
|45| 2024-03-24 | Exploring Graph Neural Networks using PyG: Working on understanding and implementing [Recurrent GNNs](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/08_RecGNN.ipynb) | GNN |
|44| 2024-03-22 | Exploring Graph Neural Networks using PyG: [Data handling in PyG](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/Data%20Handling%20in%20PyG%202.ipynb), [MetaPath2vec](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/MetaPath2Vec.ipynb) & [Graph Pooling - DiffPool](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/DIFFPOOL.ipynb)  | GNN |
|43| 2024-03-21 | Exploring Graph Neural Networks using PyG: [Edge analysis for label prediction](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/11_Node2Vec_for_label_prediction.ipynb) & [Edge analysis for link prediction](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/10_GAE_for_link_prediction.ipynb)  | GNN |
|42| 2024-03-20 | Exploring Graph Neural Networks using PyG: Graph Generation, Recurrent GNNs, [DeepWalk and Node2Vec](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/09_DeepWalk_and_node2vec.ipynb) | GNN |
|41| 2024-03-19 | Exploring Graph Neural Networks using PyG: [Spectral Graph Convolutional Layers](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/04_Convolutional_Layers_Spectral_methods.ipynb), [Aggregation Functions in GNNs](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/05_Aggregation_PNA%2BLAF.ipynb), [GAE and VGAE](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/06_GAE_VGAE.ipynb), [ARGA and ARGVA](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/07_ARGA_%26_ARVGA.ipynb) | GNN |
|40| 2024-03-18 | Exploring Graph Neural Networks using PyG: [node classification](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/02_Node_classification.ipynb) and [graph classification](https://github.com/Ramsi-K/3D-Vision-Playground/blob/main/GNN/03_Graph_classification.ipynb) tasks | GNN |
|39| 2024-03-17 | LeetCode: [0016-3sum-closest](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0016-3sum-closest) and [0017-letter-combinations-of-a-phone-number](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/0017-letter-combinations-of-a-phone-number) | DSA |
|38| 2024-03-15 | Exploring 3D object detection by implementing a model using methods including [Frustum PointNets](https://arxiv.org/abs/1711.08488) and [VoteNet](https://arxiv.org/abs/1904.09664) | DL 3D |
|37| 2024-03-14 |  Finished implementing the ESRGAN paper to [code in PyTorch](https://github.com/Ramsi-K/paper-to-code/tree/main/ESRGAN). | GANs |
|36| 2024-03-13 | Working on image super-resolution and implementing a SOTA model like [ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks)](https://arxiv.org/abs/1809.00219). GitHub Repo: [ESRGAN](https://github.com/xinntao/ESRGAN) | GANs |
|35| 2024-03-12 |  Finished implementing the PointNet paper to [code in PyTorch](https://github.com/Ramsi-K/paper-to-code/tree/main/PointNet). | DL 3D |
|34| 2024-03-11 | LeetCode problems: [15-3sum](https://github.com/Ramsi-K/python-projects/tree/main/LeetCode-solutions/15-3sum) | DSA |
|33| 2024-03-08 | Implementing the PointNet [paper](https://arxiv.org/abs/1612.00593) to [code in PyTorch](https://github.com/Ramsi-K/paper-to-code/tree/main/PointNet).| DL 3D |
|32| 2024-03-07 |Explored [PyTorch3D tutorials](https://github.com/Ramsi-K/3D-Vision-Playground/tree/main/Pytorch3D) and updated the  [3D Vision Playground](https://github.com/Ramsi-K/3D-Vision-Playground) repo. | DL 3D |
|31| 2024-03-06 |Researching [PointNet](https://github.com/Ramsi-K/paper-to-code/tree/main/PointNet) paper for code recreation | DL 3D |
|30| 2024-03-05|Implemented the [VAE paper from scratch](https://github.com/Ramsi-K/paper-to-code/tree/main/VAE) in Pytorch training on MNIST | GANs |
|29| 2024-03-04|Completed [VQGAN implementation](https://github.com/Ramsi-K/paper-to-code/tree/main/VQGAN) for code repository| GANs |
|28| 2024-03-01|Exploring the [Mesa](https://github.com/projectmesa/mesa) library for agent-based modeling, analysis and visualization | RL |
|27| 2024-02-29| Implementing [VQGAN paper](https://arxiv.org/abs/2012.09841) from [scratch in PyTorch](https://github.com/Ramsi-K/paper-to-code/tree/main/VQGAN). VQGAN debugging and scripting for transformer| GANs|
|26| 2024-02-28| Implementing [VQGAN paper](https://arxiv.org/abs/2012.09841) from [scratch in PyTorch](https://github.com/Ramsi-K/paper-to-code/tree/main/VQGAN). Scripts for encoder-decoder as well as VQGAN arch.| GANs|
|25| 2024-02-27| Built scripts for editing person's clothes in image using pretrained segmentation and diffusion models: [1](https://github.com/Ramsi-K/GANs/blob/main/DiffusionModels_InPainting.ipynb) [2](https://github.com/Ramsi-K/GANs/blob/main/Diffusion%2BClipSeg_InPainting.ipynb) | Diffusion CLIP |
|24| 2024-02-26| Implementing [VQGAN paper](https://arxiv.org/abs/2012.09841) from scratch. Understanding the paper and code repo, building skeleton.| GANs|
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
