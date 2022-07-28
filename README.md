# Domain-generalization
This is a repository for organizing papers ,codes, and etc related to **Domain Generalization**.

# Table of Contents
- [Papers](#papers)
    - [Survey papers](#Survey-papers)
    - [Theory & Analysis](#theory--analysis)
    - [Multiple Domain Generalization](#multiple-domain-generalization)
      - [Data manipulation](#domain-mainpulation)
        - [Data Augmentation-Based Methods](#data-augmentation-based-methods)
        - [Data Generation-Based Methods](#data-generation-based-methods)
      - [Representation learning](#representation-learning)
        - [Domain-Invariant Representation-Based Methods](#Domain-invariant-representation-based-methods)
        - [Disentangled Representation Learning-Based Methods](#disentangled-representation-learning-based-methods)
      - [Learning strategy](#learning-startegy)
        - [Ensemble Learning-Based Methods](#ensemble-learning-based-methods)
        - [Meta-Learning-Based Methods](#meta-learning-based-methods)
        - [Gradient Operation-Based Methods](#gradient-operation-based-methods)
        - [Regularization-Based Methods](#regularization-based-methods)
        - [Normalization-Based Methods](#normalization-based-methods)
        - [Causality-Based Methods](#causality-based-methods)
        - [Information-Based Methods](#information-based-methods)
        - [Test-Time-Based Methods](#test-time-based-methods)
      - [Others](#others)
    - [Single Domain Generalization](#single-domain-generalization)
    - [Self-Supervised Domain Generalization](#self-supervised-domain-generalization)
    - [Semi/Weak/Un-Supervised Domain Generalization](#semiweakun-supervised-domain-generalization)
    - [Open/Heterogeneous Domain Generalization](#openheterogeneous-domain-generalization)
    - [Federated Domain Generalization](#federated-domain-generalization)
- [Datasets](#datasets)
- [Libraries](#libraries)
- [Other Resources](#other-resources)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
# Papers (ongoing)

## Survey papers
- 计算机视觉领域的DG综述：   
[Domain Generalization in Vision: A Survey](https://arxiv.org/abs/2103.02503)  
Author： Zhou, Kaiyang, Ziwei Liu, Yu Qiao, Tao Xiang, and Chen Change Loy  
*arXiv preprint arXiv:2103.02503* (2021)


- OOD综述：   
[Towards Out-Of-Distribution Generalization: A Survey](https://arxiv.org/abs/2108.13624)  
Author：Zheyan Shen, Jiashuo Liu, Yue He, Xingxuan Zhang, Renzhe Xu, Han Yu, Peng Cui  
*arXiv preprint arXiv:2108.13624* (2021)   
[[Paper list]](https://out-of-distribution-generalization.com)


- DG综述：   
[Generalizing to Unseen Domains: A Survey on Domain Generalization](https://arxiv.53yu.com/pdf/2103.03097)  
Author：Wang, Jindong, Cuiling Lan, Chang Liu, Yidong Ouyang, Wenjun Zeng, and Tao Qin  
*International Joint Conference on Artificial Intelligence* (**IJCAI**) (2021)  
[[Slides]](http://jd92.wang/assets/files/DGSurvey-ppt.pdf)  [[Video]](https://www.zhihu.com/zvideo/1406391305577074688)

## Theory & Analysis
> We list the papers that either provide inspiring theoretical analyses or conduct extensive empirical studies for domain generalization.
- 研究了一个基于核的学习算法，并建立了一个泛化误差边界对DG多分类进行理论分析：  
[Explainable Deep Classification Models for Domain Generalization](https://arxiv.org/abs/1905.10392)  
 Author：Aniket Anand Deshmukh, Yunwen Lei, Srinagesh Sharma, Urun Dogan, James W. Cutler, Clayton Scott  
*arXiv preprint arXiv:1905.10392* (2019)  


- 介绍了不变风险最小化(*IRM*)，一种估计跨越多个训练分布的不变相关的学习范式：  
[Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)  
 Author：Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, David Lopez-Paz  
*arXiv preprint arXiv:1907.02893* (2019)


- 探讨Invariant Risk Minimization(*IRM*)存在的问题：  
[The Risks of Invariant Risk Minimization](https://arxiv.org/abs/2010.05761)  
 Author：Elan Rosenfeld, Pradeep Ravikumar, Andrej Risteski  
*arXiv preprint arXiv:2010.05761* (2020)


- 探讨域泛化算法在现实环境中的作用，并提出了域泛化测试平台DomainBed：  
[In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434)  
 Author：Ishaan Gulrajani, David Lopez-Paz  
*arXiv preprint arXiv:2007.01434* (2020)   
[[DomainBed]](https://github.com/facebookresearch/DomainBed)


- 用一个自动指标和人类的判断来量化图像分类模型域泛化的可解释性：  
[Explainable Deep Classification Models for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Zunino_Explainable_Deep_Classification_Models_for_Domain_Generalization_CVPRW_2021_paper.html)  
 Author：Andrea Zunino, Sarah Adel Bargal, Riccardo Volpi, Mehrnoosh Sameki, Jianming Zhang, Stan Sclaroff, Vittorio Murino, Kate Saenko  
*Conference on Computer Vision and Pattern Recognition Workshops* (**CVPR Workshops**) (2021)  


- 正式定义了在领域泛化中可以量化和计算的转移性：  
[Quantifying and Improving Transferability in Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/5adaacd4531b78ff8b5cedfe3f4d5212-Abstract.html)  
 Author：Guojun Zhang, Han Zhao, Yaoliang Yu, Pascal Poupart  
*Advances in Neural Information Processing Systems 34* (**NeurIPS**) (2021)
[[Code]](https://github.com/Gordon-Guojun-Zhang/Transferability-NeurIPS2021)


- (**OoD-Bench**) 确定并测量了在各种数据集中无处不在的两种不同类型的分布偏移，并通过广泛实验揭示了它们在一种偏移上的优势以及在另一种偏移上的局限性：  
[OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Ye_OoD-Bench_Quantifying_and_Understanding_Two_Dimensions_of_Out-of-Distribution_Generalization_CVPR_2022_paper.html)  
 Author：Nanyang Ye, Kaican Li, Haoyue Bai, Runpeng Yu, Lanqing Hong, Fengwei Zhou, Zhenguo Li, Jun Zhu  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)  
[[Code]](https://github.com/ynysjtu/ood_bench)


- 研究Domain Adversarial Neural Networks (*DANN*) 在领域泛化中的有效性：  
[Domain Adversarial Neural Networks for Domain Generalization: When It Works and How to Improve](https://arxiv.org/abs/2102.03924)  
 Author：Anthony Sicilia, Xingchen Zhao, Seong Jae Hwang  
*arXiv preprint arXiv:2102.03924* (2022)


- 现有的DG评估方法未能揭示导致性能不佳的各种因素的影响，本文提出了一个领域泛化算法的评估框架：  
[Failure Modes of Domain Generalization Algorithms](https://openaccess.thecvf.com/content/CVPR2022/html/Galstyan_Failure_Modes_of_Domain_Generalization_Algorithms_CVPR_2022_paper.html)  
 Author：Tigran Galstyan, Hrayr Harutyunyan, Hrant Khachatrian, Greg Ver Steeg, Aram Galstyan  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)

  

## Multiple Domain Generalization
> Multiple Domain Generalization aims to learn a model from multiple source domains that will generalize well on unseen target domains.
### Data manipulation
#### Data Augmentation-Based Methods
> Data augmentation-based methods augment original data to enhance the generalization performance of the model, typical augmentation operations include flipping, rotation, scaling, cropping, adding noise, and so on.
- (**CROSSGRAD**) 保留并利用了domain信息，利用domain信息辅助样本扩增，丰富了域内的数据样本：  
[Generalizing Across Domains via Cross-Gradient Training](https://openreview.net/pdf?id=r1Dx7fbCW)  
Author：Shankar, Shiv, Vihari Piratla, Soumen Chakrabarti, Siddhartha Chaudhuri, Preethi Jyothi, Sunita Sarawagi  
*International Conference on Learning Representations* (**ICLR**) (2018)   
[[Code]](https://github.com/vihari/crossgrad)


- 提出了一个迭代程序，用来自一个虚构的目标域的例子来扩充源域数据：   
[Generalizing to Unseen Domains via Adversarial Data Augmentation](https://proceedings.neurips.cc/paper/2018/hash/1d94108e907bb8311d8802b48fd54b4a-Abstract.html)  
 Author：Riccardo Volpi, Hongseok Namkoong, Ozan Sener, John C. Duchi, Vittorio Murino, Silvio Savarese  
*Advances in Neural Information Processing Systems 31* (**NeurIPS**) (2018)  
[[Code]](https://github.com/ricvolpi/generalize-unseen-domains)


- (**JiGen**) 以监督的方式学习语义标签，并通过从自我监督的信号中学习如何解决相同图像上的拼图来提升泛化能力：  
[Domain Generalization by Solving Jigsaw Puzzles](https://openaccess.thecvf.com/content_CVPR_2019/html/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.html)  
 Author：Fabio M. Carlucci, Antonio D'Innocente, Silvia Bucci, Barbara Caputo, Tatiana Tommasi  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2019)  
[[Code]](https://github.com/fmcarlucci/JigenDG)


- (**DRPC**) 提出了一种新的域随机化和金字塔一致性的方法，以学习一个具有高泛化能力的模型：  
[Domain Randomization and Pyramid Consistency: Simulation-to-Real Generalization Without Accessing Target Domain Data](https://openaccess.thecvf.com/content_ICCV_2019/html/Yue_Domain_Randomization_and_Pyramid_Consistency_Simulation-to-Real_Generalization_Without_Accessing_Target_ICCV_2019_paper.html)  
 Author：Xiangyu Yue, Yang Zhang, Sicheng Zhao, Alberto Sangiovanni-Vincentelli, Kurt Keutzer, Boqing Gong  
*International Conference on Computer Vision* (**ICCV**) (2019)  
[[Code]](https://github.com/xyyue/DRPC)


- (**M-ADA**) 提出了一种名为对抗性领域增强的新方法来来创建 "虚构 "而又 "具有挑战性 "的样本，进而解决分布外（*OOD*）的泛化问题：  
[Learning to Learn Single Domain Generalization](https://openaccess.thecvf.com/content_CVPR_2020/html/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.html)  
 Author：Fengchun Qiao, Long Zhao, Xi Peng  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2020)  
[[Code]](https://github.com/joffery/M-ADA)


- (**EISNet**) 提出了一个新的领域泛化框架（称为EISNet），利用多任务学习范式，从多源领域的图像的外在关系监督和内在自我监督中同时学习如何跨领域泛化：   
[Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-030-58545-7_10)  
 Author：Shujun Wang, Lequan Yu, Caizi Li, Chi-Wing Fu, Pheng-Ann Heng   
*Proceedings of the European Conference on Computer Vision* (**ECCV**) (2020)  
[[code]](https://github.com/EmmaW8/EISNet)


- (**DecAug**) ：提出了一种新颖的分解特征表示和语义增强的方法，用于OoD泛化   
[DecAug: Out-of-Distribution Generalization via Decomposed Feature Representation and Semantic Augmentation](https://ojs.aaai.org/index.php/AAAI/article/view/16829)  
Author： Haoyue Bai, Rui Sun, Lanqing Hong, Fengwei Zhou, Nanyang Ye, Han-Jia Ye, S.-H. Gary Chan, Zhenguo Li      
*Association for the Advancement of Artificial Intelligence* (**AAAI**) (2021)


- (**MixStyle**) 本文提出了一种基于概率地混合源域中训练样本的实例级特征统计的方法。混合训练实例的风格导致新的领域被隐含地合成，这增加了源领域的多样性，从而增加了训练模型的泛化能力：  
[Domain Generalization with MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp)  
 Author：Kaiyang Zhou, Yongxin Yang, Yu Qiao, Tao Xiang  
*International Conference on Learning Representations* (**ICLR**) (2021)  
[[Code]](https://github.com/KaiyangZhou/mixstyle-release)


- (**FSDR**) 提出了频率空间域随机化（*FSDR*），通过保留域不变的频率分量（*DIFs*）和只随机化域可变的频率分量（*DVFs*），在频率空间中随机化图像：  
[FSDR: Frequency Space Domain Randomization for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_FSDR_Frequency_Space_Domain_Randomization_for_Domain_Generalization_CVPR_2021_paper.html)    
 Author：Huang, Jiaxing, Dayan Guan, Aoran Xiao, and Shijian Lu   
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)


- (**ATSRL**) 多视角学习，提出对抗性师生表征学习框架，将表征学习和数据增广相结合，前者逐步更新教师网络以得出域通用的表征，而后者则合成数据的外源但合理的分布：  
[Adversarial Teacher-Student Representation Learning for Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/a2137a2ae8e39b5002a3f8909ecb88fe-Abstract.html)  
 Author：Fu-En Yang, Yuan-Chia Cheng, Zu-Yun Shiau, Yu-Chiang Frank Wang  
*Advances in Neural Information Processing Systems 34* (**NeurIPS**) (2021)  


- (**MBDG**) 提出了一种具有收敛保证的新型域泛化算法：   
[Model-Based Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/a8f12d9486cbcc2fe0cfc5352011ad35-Abstract.html)   
 Author：Alexander Robey, George J. Pappas, Hamed Hassani   
*Advances in Neural Information Processing Systems 34* (**NeurIPS**) (2021)   
[[Code]](https://github.com/arobey1/mbdg)


- (**FACT**) 开发了一种新颖的基于傅里叶的数据增强策略，并引入了一种称为co-teacher regularization的双重形式的一致性损失来学习域不变表征：  
[A Fourier-Based Framework for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.html)  
 Author：Qinwei Xu, Ruipeng Zhang, Ya Zhang, Yanfeng Wang, Qi Tian  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)


- (**SBL**) 开发了一种多视图正则化的元学习算法，在更新模型时采用多个任务来产生合适的优化方向。在测试阶段，利用多个增强的图像来产生多视图预测，通过融合测试图像的不同视图的结果来显著提高模型的可靠性：  
[More is Better: A Novel Multi-view Framework for Domain Generalization](https://arxiv.org/abs/2112.12329)  
 Author：Jian Zhang, Lei Qi, Yinghuan Shi, Yang Gao  
*arXiv preprint arXiv:2112.12329* (2021)


- 这项工作奠定了领域泛化的学习理论基础，提出了两个正式的数据生成模型，相应的风险概念，以及无分布泛化误差分析：  
[Domain Generalization by Marginal Transfer Learning](https://dl.acm.org/doi/abs/10.5555/3546258.3546260)   
Author：Blanchard, Gilles, Aniket Anand Deshmukh, Urun Dogan, Gyemin Lee, and Clayton Scott  
*Journal of Machine Learning Research* (**JMLR**) (2021)


- (**FSR**) 开发了一个简单而有效的基于特征的风格随机化模块来实现特征级别的增强，通过将随机噪声集成到原始风格中来产生随机风格：  
[Feature-based Style Randomization for Domain Generalization](https://ieeexplore.ieee.org/abstract/document/9716108)  
 Author：Yue Wang, Lei Qi, Yinghuan Shi, Yang Gao  
*IEEE Transactions on Circuits and Systems for Video Technology* (**TCSVT CCFB**) (2022)


- (**Style Neophile**) 通过不断地产生新的和合理的风格，并用合成的风格来增强训练图像：  
[Style Neophile: Constantly Seeking Novel Styles for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Kang_Style_Neophile_Constantly_Seeking_Novel_Styles_for_Domain_Generalization_CVPR_2022_paper.html)  
Author：Juwon Kang, Sohyun Lee, Namyup Kim, Suha Kwak    
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)  


- (**EFDM**) 将任意风格转移 (*AST*) 和领域泛化 (*DG*) 相结合：  
[Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Exact_Feature_Distribution_Matching_for_Arbitrary_Style_Transfer_and_Domain_CVPR_2022_paper.html)  
 Author：Yabin Zhang, Minghan Li, Ruihuang Li, Kui Jia, Lei Zhang  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)  
[[Code]](https://github.com/YBZh/EFDM)


- (**DDG**) 将OOD泛化问题形式化为约束性优化问题，称为Disentanglement-constrained Domain Generalization (*DDG*)：  
[Towards Principled Disentanglement for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Towards_Principled_Disentanglement_for_Domain_Generalization_CVPR_2022_paper.html)    
 Author：Hanlin Zhang, Yi-Fan Zhang, Weiyang Liu, Adrian Weller, Bernhard Schölkopf, Eric P. Xing  
*Conference on Computer Vision and Pattern Recognition* (**CVPR Oral**) (2022)  
[[code]](https://github.com/hlzhang109/DDG)

  
#### Data Generation-Based Methods
> Data generation-based methods aim to generate new domain samples using some generative models such as Generative Adversarial Networks (GAN), Variational Auto-encoder (VAE).
- (**UFDN**) 过对抗性训练和利用特定域信息的额外能力来学习域不变的表征：   
[A Unified Feature Disentangler for Multi-Domain Image Translation and Manipulation](https://proceedings.neurips.cc/paper/2018/hash/84438b7aae55a0638073ef798e50b4ef-Abstract.html)  
Author：Alexander H. Liu, Yen-Cheng Liu, Yu-Ying Yeh, Yu-Chiang Frank Wang  
*Neural Information Processing Systems 31* (**NeurIPS**) (2018)    
[[Code]](https://github.com/Alexander-H-Liu/UFDN)


- (**ADAGE**) 提出了第一个端到端的图像和特征级联合自适应的DG解决方案，在像素和特征层面的两个对抗性自适应条件的指导下幻化出领域不可知的图像：   
[Hallucinating Agnostic Images to Generalize Across Domains](https://ieeexplore.ieee.org/abstract/document/9022393)  
Author：Fabio Maria Carlucci, Paolo Russo, Tatiana Tommasi, Barbara Caputo  
*International Conference on Computer Vision Workshops* (**ICCV Workshps**) (2019)


- (**MIT-DG**) 以两种方式解决域泛化问题，一是利用生成对抗网络(*GAN*)生成的合成数据，二是介绍了一种将DA方法应用于DG场景的协议：   
[Multi-component Image Translation for Deep Domain Generalization](https://ieeexplore.ieee.org/abstract/document/8658643)  
Author：Mohammad Mahfujur Rahman, Clinton Fookes, Mahsa Baktashmotlagh, Sridha Sridharan  
*IEEE Workshop on Applications of Computer Vision* (**WACV**) (2019)    
[[Code]](https://github.com/mahfujur1/mit-DG)


- (**DLOW**) 通过生成一个从一个域流向另一个域的连续的中间域序列来连接两个不同的域，弥合了源域和目标域之间的差距：   
[DLOW: Domain Flow for Adaptation and Generalization](https://openaccess.thecvf.com/content_CVPR_2019/html/Gong_DLOW_Domain_Flow_for_Adaptation_and_Generalization_CVPR_2019_paper.html)  
Author：Rui Gong, Wen Li, Yuhua Chen, Luc Van Gool  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2019)    
[[Code]](https://github.com/ETHRuiGong/DLOW)


- (**FSDG**) 通过用风格化的图像增加数据集来纠正域偏移：   
[Frustratingly Simple Domain Generalization via Image Stylization](https://arxiv.org/abs/2006.11207)     
Author：Somavarapu, Nathan, Chih-Yao Ma, and Zsolt Kira    
*arXiv preprint arXiv:2006.11207* (2020)    
[[code]](https://github.com/GT-RIPL/DomainGeneralization-Stylization)


- 提出了一种新的异质域泛化方法，即用两种不同的采样策略将多个源域的样本混合起来：  
[Heterogeneous Domain Generalization Via Domain Mixup](https://ieeexplore.ieee.org/abstract/document/9053273)     
Author：Yufei Wang, Haoliang Li, Alex C. Kot    
*IEEE International Conference on Acoustics, Speech and Signal Processing* (**ICASSP**) (2020)


- (**L2A-OT**) 采用了一个数据生成器来合成伪的新领域的数据，以增强源领域的能力：  
[Learning to Generate Novel Domains for Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-030-58517-4_33)  
 Author：Kaiyang Zhou, Yongxin Yang, Timothy Hospedales，Tao Xiang    
*Proceedings of the European Conference on Computer Vision* (**ECCV**) (2020)  
[[code]](https://github.com/EmmaW8/EISNet)


- (**DDAIG**) 提出了一种基于深度域对抗性图像生成(*DDAIG*)的新型DG方法：  
[Deep Domain-Adversarial Image Generation for Domain Generalisation](https://ojs.aaai.org/index.php/AAAI/article/view/7003)  
 Author：Kaiyang Zhou, Yongxin Yang, Timothy Hospedales，Tao Xiang
*Association for the Advancement of Artificial Intelligence* (**AAAI**) (2020)  
[[code]](https://github.com/KaiyangZhou/Dassl.pytorch)


- (**M^3L**) 引入了一个元学习策略来模拟训练-测试的过程，还提出了一个元批量规范化层(*MetaBN*)来使元测试特征多样化：   
[Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification](https://openaccess.thecvf.com/content/CVPR2021/html/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.html)  
Author：Yuyang Zhao, Zhun Zhong, Fengxiang Yang, Zhiming Luo, Yaojin Lin, Shaozi Li, Nicu Sebe  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)    
[[Code]](https://github.com/HeliosZhao/M3L)


- (**PDEN**) 提出了一个新颖的渐进式域扩展网络 (*PDEN*) 学习框架，通过逐渐生成模拟目标与数据，提升模型泛化能力：   
[Progressive Domain Expansion Network for Single Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.html)  
Author：Fengchun Qiao, Xi Peng  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)    
[[Code]](https://github.com/lileicv/PDEN)


- (**semanticGAN**) 训练生成式对抗网络以捕捉图像-标签的联合分布，并使用大量的未标记图像和少量的标记图像进行有效的训练：   
[Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Semantic_Segmentation_With_Generative_Models_Semi-Supervised_Learning_and_Strong_Out-of-Domain_CVPR_2021_paper.html)  
Author：Daiqing Li, Junlin Yang, Karsten Kreis, Antonio Torralba, Sanja Fidler  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)


- (**RICE**) 数据生成与因果学习结合，基于修改非因果特征但不改变因果部分的转换，在不明确恢复因果特征的情况下解决OOD问题：  
[Out-of-Distribution Generalization With Causal Invariant Transformations](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Out-of-Distribution_Generalization_With_Causal_Invariant_Transformations_CVPR_2022_paper.html)  
 Author：Ruoyu Wang, Mingyang Yi, Zhitang Chen, Shengyu Zhu  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)  



### Representation learning
#### Domain-Invariant Representation-Based Methods
>Domain-invariant representation-based methods aim to reduce the representation discrepancy between multiple source domains in a specific feature space to be domain invariant so that the learned model can have a generalizable capability to the unseen domain.
- (**DICA**) 基于核的优化算法，通过最小化跨域的不相似性来学习不变的转换：  
[Domain generalization via invariant feature representation](https://proceedings.mlr.press/v28/muandet13.html)   
Author：Muandet, Krikamol, David Balduzzi, and Bernhard Schölkopf   
*International Conference on Machine Learning* (**ICML**) (2013)  
[[code]](https://github.com/krikamol/dg-dica)


- (**MTAE**) 提出了一种新的特征学习算法--多任务自动编码器 (*MTAE*)，通过学习将原始图像转化为多个相关域中的类似物：  
[Domain Generalization for Object Recognition With Multi-Task Autoencoders](https://openaccess.thecvf.com/content_iccv_2015/html/Ghifary_Domain_Generalization_for_ICCV_2015_paper.html)   
Author：Muhammad Ghifary, W. Bastiaan Kleijn, Mengjie Zhang, David Balduzzi   
*International Conference on Computer Vision* (**ICCV**) (2015)  


- (**KDICA**) 为多源领域泛化来开发一种新的面向属性的特征表示，以方便应用现成的分类器来获得高质量的属性检测器：   
[Learning Attributes Equals Multi-Source Domain Generalization](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gan_Learning_Attributes_Equals_CVPR_2016_paper.html)   
Author：Chuang Gan, Tianbao Yang, Boqing Gong   
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2016)  


- (**ESRand**) 通过减少多领域学习中的分布偏差学习域不变表征，提升模型泛化能力：   
[Robust domain generalisation by enforcing distribution invariance](https://eprints.qut.edu.au/115382/)   
Author：Erfani, Sarah, Baktashmotlagh, Mahsa, Moshtaghi, Masud, Nguyen, Xuan, Leckie, Christopher, Bailey, James, Kotagiri, Rao   
*International Joint Conference on Artificial Intelligence 25* (**IJCAI**) (2016) 


- (**SCA**) 提出了Scatter Component Analyis (*SCA*)，最大限度地提高类别的可分离性、最小化领域之间的不匹配以及最大限度地提高数据的可分离性：
[Scatter Component Analysis: A Unified Framework for Domain Adaptation and Domain Generalization](https://ieeexplore.ieee.org/abstract/document/7542175)   
Author：Muhammad Ghifary, David Balduzzi, W. Bastiaan Kleijn, Mengjie Zhang   
*IEEE Transactions on Pattern Analysis and Machine Intelligence* (**TPAMI CCF-A**) (2017)  


- (**CCSA**) 利用连体结构与对比性损失来解决领域转换和泛化问题：  
[Unified Deep Supervised Domain Adaptation and Generalization](https://openaccess.thecvf.com/content_iccv_2017/html/Motiian_Unified_Deep_Supervised_ICCV_2017_paper.html)   
Author：Saeid Motiian, Marco Piccirilli, Donald A. Adjeroh, Gianfranco Doretto   
*International Conference on Computer Vision* (**ICCV**) (2017)  

  
- (**CIDDG**) 提出了一个端到端的条件不变的深度域泛化方法，利用深度神经网络进行领域不变的表征学习：  
[Deep Domain Generalization via Conditional Invariant Adversarial Networks](https://openaccess.thecvf.com/content_ECCV_2018/html/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.html)  
 Author：Ya Li, Xinmei Tian, Mingming Gong, Yajing Liu, Tongliang Liu, Kun Zhang, Dacheng Tao  
*European Conference on Computer Vision* (**ECCV**) (2018)


- (**MMD-AAE**) 提出了一个基于对抗性自动编码器的新框架，使不同领域的分布一致以学习跨领域的广义潜在特征表示：  
[Domain Generalization With Adversarial Feature Learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Domain_Generalization_With_CVPR_2018_paper.html)  
 Author：Haoliang Li, Sinno Jialin Pan, Shiqi Wang, Alex C. Kot  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2018)


- (**CIDG**) 提出了一种条件不变的域泛化方法，考虑到P(X)和P(Y|X)都会跨域变化的情况：  
[Domain Generalization via Conditional Invariant Representations](https://ojs.aaai.org/index.php/AAAI/article/view/11682)  
 Author：Ya Li, Mingming Gong, Xinmei Tian, Tongliang Liu, Dacheng Tao  
*Association for the Advancement of Artificial Intelligence* (**AAAI**) (2018)


- (**G2DM**) 采用了多个一比一的领域判别器，从而在训练时估计并最小化源分布之间的配对分歧：   
[Generalizing to unseen domains via distribution matching](https://arxiv.org/abs/1911.00804)  
 Author：Isabela Albuquerque, João Monteiro, Mohammad Darvishi, Tiago H. Falk, Ioannis Mitliagkas  
*arXiv preprint arXiv:1911.00804* (2019)


- (**MDA**) 提出了多域判别分析 (*MDA*) 学习一个领域不变的特征转换：  
[Domain Generalization via Multidomain Discriminant Analysis](https://proceedings.mlr.press/v115/hu20a.html)    
Author：Hu, Shoubo, Kun Zhang, Zhitang Chen, Laiwan Chan  
*Conference on Uncertainty in Artificial Intelligence* (**PMLR-UAI**) 2019  
[[code]](https://github.com/amber0309/Multidomain-Discriminant-Analysis)


- (**G2DM**) 采用了多个一比一的领域判别器，从而在训练时估计并最小化源分布之间的配对分歧：   
[Generalizing to unseen domains via distribution matching](https://arxiv.org/abs/1911.00804)  
 Author：Isabela Albuquerque, João Monteiro, Mohammad Darvishi, Tiago H. Falk, Ioannis Mitliagkas  
*arXiv preprint arXiv:1911.00804* (2019)


- (**MMLD**) 介绍了使用多个潜在域的混合用于领域泛化，作为一种新的和更现实的场景，其试图在不使用域标签的情况下训练一个域泛化的模型：  
[Domain Generalization Using a Mixture of Multiple Latent Domains](https://ojs.aaai.org/index.php/AAAI/article/view/6846)  
 Author：Toshihiko Matsuura, Tatsuya Harada  
*Association for the Advancement of Artificial Intelligence* (**AAAI**) (2020)   
[[Code]](https://github.com/mil-tokyo/dg_mmld)

  
- (**BNE**) 依靠特定领域的归一化层来分解每个训练领域的独立表征，然后使用这种隐式嵌入来定位来自未知域的未见过的样本：   
[Batch Normalization Embeddings for Deep Domain Generalization](https://arxiv.org/abs/2011.12672)  
Author： Mattia Segu, Alessio Tonioni, Federico Tombari   
*arXiv preprint arXiv:2011.12672* (2020)


- (**ZSDG**) 将DG扩展到一个更具挑战性的环境中，即未见过的领域的标签空间也可能发生变化：   
[Zero Shot Domain Generalization](https://arxiv.org/abs/2008.07443)   
Author： Udit Maniyar, Joseph K J, Aniket Anand Deshmukh, Urun Dogan, Vineeth N Balasubramanian  
*British Machine Vision Conference* (**BMVC**) (2020) 

  
- (**SFA**) 提出了一种基于特征增强的增强方法，即在训练过程中用高斯噪声扰动特征嵌入对源域数据进行增广：  
[A Simple Feature Augmentation for Domain Generalization](https://openaccess.thecvf.com/content/ICCV2021/html/Li_A_Simple_Feature_Augmentation_for_Domain_Generalization_ICCV_2021_paper.html)  
 Author：Pan Li, Da Li, Wei Li, Shaogang Gong, Yanwei Fu, Timothy M. Hospedales  
*International Conference on Computer Vision* (**ICCV**) (2021)


- (**DFDG**) ：在不需要源域标签的情况下，通过类别条件的软标签来协调样本的类别关系，以学习领域不变的类区分特征：  
[Robust Domain-Free Domain Generalization with Class-Aware Alignment](https://ieeexplore.ieee.org/abstract/document/9413872)  
 Author：Wenyu Zhang; Mohamed Ragab; Ramon Sagarna  
*International Conference on Acoustics, Speech, and Signal Processing* (**ICASSP CCF-B**) (2021)


- (**ATSRL**) 多视角学习，提出对抗性师生表征学习框架，将表征学习和数据增广相结合，前者逐步更新教师网络以得出域通用的表征，而后者则合成数据的外源但合理的分布：  
[Adversarial Teacher-Student Representation Learning for Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/a2137a2ae8e39b5002a3f8909ecb88fe-Abstract.html)  
 Author：Fu-En Yang, Yuan-Chia Cheng, Zu-Yun Shiau, Yu-Chiang Frank Wang  
*Advances in Neural Information Processing Systems 34* (**NeurIPS**) (2021)  


- (**LADG**) 提出了具有空间紧凑性维护的局部对抗式域泛化 (*LADG*)，解决了以往对抗式域泛化的限制：  
[Localized Adversarial Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Localized_Adversarial_Domain_Generalization_CVPR_2022_paper.html)  
 Author：Wei Zhu, Le Lu, Jing Xiao, Mei Han, Jiebo Luo, Adam P. Harrison  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)

  
- (**BatchFormer**) 引入了一个BatchFormer模块，将其应用于每个mini-batch的批处理维度，在训练期间隐含地探索样本关系：  
[BatchFormer: Learning To Explore Sample Relationships for Robust Representation Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Hou_BatchFormer_Learning_To_Explore_Sample_Relationships_for_Robust_Representation_Learning_CVPR_2022_paper.html)  
 Author：Zhi Hou, Baosheng Yu, Dacheng Tao  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)


#### Disentangled Representation Learning-Based Methods
>Disentangled representation learning-based methods aim to disentangle domain-specific and domain-invariant parts from source data, and then adopt the domain-invariant one for inference on the target domains.
- (**Undo-Bias**) 提出了一个鉴别性的框架，在训练中直接利用数据集的偏差：  
[Undoing the damage of dataset bias](https://link.springer.com/chapter/10.1007/978-3-642-33718-5_12)   
Author：Khosla, Aditya, Tinghui Zhou, Tomasz Malisiewicz, Alexei A. Efros, Antonio Torralba   
*European Conference on Computer Vision* (**ECCV**) (2012)   
[[code]](https://github.com/adikhosla/undoing-bias)


- 为端到端DG学习开发了一个低秩参数化的CNN模型，其次提出了一个新的DG数据集——PACS，具有更大的域偏移：  
[Deeper, broader and artier domain generalization](https://openaccess.thecvf.com/content_iccv_2017/html/Li_Deeper_Broader_and_ICCV_2017_paper.html)    
Author：Zhengming Ding, Yun Fu   
*Proceedings of the IEEE International Conference on Computer Vision* (**ICCV**) (2017)   
[[code]](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)


- (**SLRC**) 开发了一个具有结构化低秩约束的深度域泛化框架，通过捕捉多个相关源领域的一致知识来促进未见过的目标域评估：  
[Deep Domain Generalization With Structured Low-Rank Constraint](https://ieeexplore.ieee.org/abstract/document/8053784)    
Author：Li, Da, Yongxin Yang, Yi-Zhe Song, Timothy M. Hospedales   
*IEEE Transactions on Image Processing* (**TIP CCF-A**) (2017)


- (**UFDN**) 过对抗性训练和利用特定域信息的额外能力来学习域不变的表征：   
[A Unified Feature Disentangler for Multi-Domain Image Translation and Manipulation](https://proceedings.neurips.cc/paper/2018/hash/84438b7aae55a0638073ef798e50b4ef-Abstract.html)  
Author：Alexander H. Liu, Yen-Cheng Liu, Yu-Ying Yeh, Yu-Chiang Frank Wang  
*Neural Information Processing Systems 31* (**NeurIPS**) (2018)    
[[Code]](https://github.com/Alexander-H-Liu/UFDN)


- (**DADA**) 提出了一种新的深度对抗性分解自动编码器 (*DADA*)来分解潜在空间中的域不变特征：  
[DIVA: Domain Invariant Variational Autoencoders](https://proceedings.mlr.press/v121/ilse20a.html)   
Author：Xingchao Peng, Zijun Huang, Ximeng Sun, Kate Saenko  
*International Conference on Machine Learning* (**PMLR-ICML**) (2019)  
[[code]](https://github.com/VisionLearningGroup/DAL)


- (**DMG**) 提出了用于泛化的特定域mask (*DMG*) 它平衡了特定领域和领域不变的特征表征，以产生一个能够有效泛化的单一强大模型：
[Learning to Balance Specificity and Invariance for In and Out of Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-030-58545-7_18)    
 Author：Chattopadhyay, Prithvijit, Yogesh Balaji, Judy Hoffman  
*European Conference on Computer Vision* (**ECCV**) (2020)  
[[code]](https://github.com/prithv1/DMG)


- 提出了一种高效的跨域人脸脸部攻击检测的分解表征学习，包括分解表征学习(DR-Net)和多域学习(MD-Net)：    
[Cross-domain Face Presentation Attack Detection via Multi-domain Disentangled Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Domain_Face_Presentation_Attack_Detection_via_Multi-Domain_Disentangled_Representation_Learning_CVPR_2020_paper.html)    
 Author：Guoqing Wang, Hu Han, Shiguang Shan, Xilin Chen  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2020)  
[[code]](https://github.com/prithv1/DMG)


- (**CSD**) 提出了Common Specific Decomposition (*CSD*)，联合学习了一个通用组件和一个特定域组件，训练后，特定域的成分被丢弃，只有共同成分被保留：  
[Efficient Domain Generalization via Common-Specific Low-Rank Decomposition](https://proceedings.mlr.press/v119/piratla20a.html)   
Author：Vihari Piratla, Praneeth Netrapalli, Sunita Sarawagi  
*International Conference on Machine Learning* (**PMLR-ICML**) (2020)  
[[code]](https://github.com/vihari/CSD)


- (**DIVA**) 提出了Domain Invariant Variational Autoencoder (DIVA)，通过学习三个独立的潜在子空间来解决DG问题：  
[DIVA: Domain Invariant Variational Autoencoders](https://proceedings.mlr.press/v121/ilse20a.html)   
Author：Vihari Piratla, Praneeth Netrapalli, Sunita Sarawagi  
*International Conference on Machine Learning* (**PMLR-ICML Workshop**) (2020)  
[[code]](https://github.com/AMLab-Amsterdam/DIVA)


- (**SNR**) 提出了一个简单而有效的风格标准化和重构 (*SNR*) 模块,通过归一化 (In-stance Normalization，IN) 过滤掉风格的变化：   
[Style Normalization and Restitution for Generalizable Person Re-Identification](https://openaccess.thecvf.com/content_CVPR_2020/html/Jin_Style_Normalization_and_Restitution_for_Generalizable_Person_Re-Identification_CVPR_2020_paper.html)  
Author： Xin Jin, Cuiling Lan, Wenjun Zeng, Zhibo Chen, Li Zhang   
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2020)


- 提出了一种新的元学习方案，该方案具有特征分解能力，它为语义分割推导出具有域泛化保证的域变量特征：  
[Meta-Learned Feature Critics for Domain Generalized Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/9506281)  
 Author：Zu-Yun Shiau, Wei-Wei Lin, Ci-Siang Lin, Yu-Chiang Frank Wang  
*IEEE International Conference on Image Processing* (**ICIP CCF-C**) (2021)


- (**DecAug**) ：提出了一种新颖的分解特征表示和语义增强的方法，用于OoD泛化：  
[DecAug: Out-of-Distribution Generalization via Decomposed Feature Representation and Semantic Augmentation](https://ojs.aaai.org/index.php/AAAI/article/view/16829)  
Author： Haoyue Bai, Rui Sun, Lanqing Hong, Fengwei Zhou, Nanyang Ye, Han-Jia Ye, S.-H. Gary Chan, Zhenguo Li      
*Association for the Advancement of Artificial Intelligence* (**AAAI**) (2021)


- (**CSG**) 提出了一个基于因果推理的因果语义生成模型 (*CSG*)，以便对语义因素和变化因素进行单独建模：  
[Learning Causal Semantic Representation for Out-of-Distribution Prediction](https://proceedings.neurips.cc/paper/2021/hash/310614fca8fb8e5491295336298c340f-Abstract.html)  
 Author：Chang Liu, Xinwei Sun, Jindong Wang, Haoyue Tang, Tao Li, Tao Qin, Wei Chen, Tie-Yan Liu  
*Neural Information Processing Systems 34* (**NeurIPS**) (2021)   
[[Code]](https://github.com/changliu00/causal-semantic-generative-model)


- (**mDSDI**) ：提出了一种mDSDI算法，可以分域特定和域不变特征，并使用元训练方案，以支持特定领域的信息从源域到未见域的适应：  
[Exploiting Domain-Specific Features to Enhance Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/b0f2ad44d26e1a6f244201fe0fd864d1-Abstract.html)  
 Author：Manh-Ha Bui, Toan Tran, Anh Tran, Dinh Phung  
*Neural Information Processing Systems 34* (**NeurIPS**) (2021)   
[[Code]](https://github.com/manhhabui/mDSDI)


- (**SNR**) 提出了一个简单而有效的风格标准化和重构 (*SNR*) 模块,通过归一化 (In-stance Normalization，IN) 过滤掉风格的变化：   
[Style Normalization and Restitution for Domain Generalization and Adaptation](https://ieeexplore.ieee.org/abstract/document/9513542)  
Author： Xin Jin, Cuiling Lan, Wenjun Zeng, Zhibo Chen   
*IEEE Transactions on Multimedia* (**TMM CCF-B**) (2021)


- (**RobustNet**) 将特定领域的风格和在特征表征的高阶统计（即特征协方差）中编码的域不变的内容分开，并有选择地只删除导致领域转移的风格信息：  
[RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening](https://openaccess.thecvf.com/content/CVPR2021/html/Choi_RobustNet_Improving_Domain_Generalization_in_Urban-Scene_Segmentation_via_Instance_Selective_CVPR_2021_paper.html)    
 Author：Chattopadhyay, Prithvijit, Yogesh Balaji, Judy Hoffman.  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)  
[[code]](https://github.com/shachoi/RobustNet)


- (**DDG**) 将OOD泛化问题形式化为约束性优化问题，称为Disentanglement-constrained Domain Generalization (*DDG*)：  
[Towards Principled Disentanglement for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Towards_Principled_Disentanglement_for_Domain_Generalization_CVPR_2022_paper.html)    
 Author：Hanlin Zhang, Yi-Fan Zhang, Weiyang Liu, Adrian Weller, Bernhard Schölkopf, Eric P. Xing  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)  
[[code]](https://github.com/hlzhang109/DDG)


### Learning strategy
> Some methods also use some machine learning paradigms to solve DG tasks.
#### Ensemble Learning-Based Methods
> Ensemble learning usually combines multiple models, such as classifiers or experts, to enhance the power of models to make accurate prediction.
- (**LRE-SVM**) 利用多个潜在源域的低秩结构来实现域的泛化：  
[Exploiting low-rank structure from latent domains for domain generalization](https://link.springer.com/chapter/10.1007/978-3-319-10578-9_41)   
Author：Xu, Zheng, Wen Li, Li Niu, and Dong Xu   
*European Conference on Computer Vision* (**ECCV**) (2014)   
[[code]](http://www.vision.ee.ethz.ch/~liwenw/papers/Xu_ECCV2014_codes.zip)


- (**MVDG**) 使用具有多种类型特征（即多视角特征）的源域样本来学习具有泛化能力的分类器：    
[Multi-view domain generalization for visual recognition](https://openaccess.thecvf.com/content_iccv_2015/html/Niu_Multi-View_Domain_Generalization_ICCV_2015_paper.html)  
Author：Niu, Li, Wen Li, and Dong Xu  
*Proceedings of the IEEE International Conference on Computer Vision* (**ICCV**) (2015)


- 设计了一个具有多个特定领域分类器的深度网络，每个分类器与一个源领域相关：     
[Best Sources Forward: Domain Generalization through Source-Specific Nets](https://ieeexplore.ieee.org/abstract/document/8451318)  
Author：Massimiliano Mancini, Samuel Rota Bulò, Barbara Caputo, Elisa Ricci    
*IEEE International Conference on Image Processing* (**ICIP CCF-C**)(2018)


- (**WBN**) ：采用了BN的加权公式来学习稳健的分类器，这些分类器可以应用于以前未见过的目标领域：     
[Robust Place Categorization With Deep Domain Generalization](https://ieeexplore.ieee.org/abstract/document/8302933)  
Author：Massimiliano Mancini, Samuel Rota Bulò, Barbara Caputo, Elisa Ricci    
*IEEE Robotics and Automation Letters* (2018)


- (**D-SAMs**) 通过引入特定领域的聚合模块来合并通用和特定信息：     
[Domain Generalization with Domain-Specific Aggregation Modules](https://link.springer.com/chapter/10.1007/978-3-030-12939-2_14)  
Author：Antonio D’Innocente, Barbara Caputo    
*German Conference on Pattern Recognition* (**GCPR**)(2018)


- (**BNE**) 依靠特定领域的归一化层来分解每个训练领域的独立表征，然后使用这种隐式嵌入来定位来自未知域的未见过的样本：   
[Batch Normalization Embeddings for Deep Domain Generalization](https://arxiv.org/abs/2011.12672)  
Author： Mattia Segu, Alessio Tonioni, Federico Tombari   
*arXiv preprint arXiv:2011.12672* (2020)


- (**DoFE**) 提出了Domain-oriented Feature Embedding (*DoFE*)框架，从多源域学到的额外域先验知识来动态地丰富图像特征，使语义特征更具辨别力：   
[DoFE: Domain-Oriented Feature Embedding for Generalizable Fundus Image Segmentation on Unseen Datasets](https://ieeexplore.ieee.org/abstract/document/9163289)  
Author： Shujun Wang, Lequan Yu, Kang Li, Xin Yang, Chi-Wing Fu, Pheng-Ann Heng   
*TEEE Transactions on Medical Imaging* (**TMI CCF-B**) (2020)


- (**MS-Net**) 提出了一种新型的多部位网络(*MS-Net*)，通过学习稳健的表征，并利用多种数据来源来改善前列腺的分割：   
[MS-Net: Multi-Site Network for Improving Prostate Segmentation With Heterogeneous MRI Data](https://ieeexplore.ieee.org/abstract/document/9000851)  
Author： Quande Liu, Qi Dou, Lequan Yu, Pheng Ann Heng   
*TEEE Transactions on Medical Imaging* (**TMI CCF-B**) (2020)


- (**GCFN**) 提出了一个广义的卷积森林网络来学习一个特征空间，以最大化单个树分类器的强度，同时最小化各自的相关性：  
[Generalized Convolutional Forest Networks for Domain Generalization and Visual Recognition](https://openreview.net/pdf?id=H1lxVyStPH)  
Author： Ryu, Jongbin, Gitaek Kwon, Ming-Hsuan Yang, Jongwoo Lim  
*International Conference on Learning Representations* (**ICLR**) 2020


- (**DSON**) 采用了多种规范化方法，为各个域的优化设计了归一化层，同时每个领域的学习采用单独的仿生参数：  
[Learning to Optimize Domain Specific Normalization for Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-030-58542-6_5)  
 Author：Seonguk Seo, Yumin Suh, Dongwan Kim, Geeho Kim, Jongwoo Han, Bohyung Han   
*European Conference on Computer Vision* (**ECCV**) (2020)


- 提出了一种稳健的领域泛化方法，可以有效地学习一个通用的分类器，不受类别条件分布的领域转移的影响：     
[Class-conditioned Domain Generalization via Wasserstein Distributional Robust Optimization](https://arxiv.org/abs/2109.03676)  
Author：Jingge Wang, Yang Li, Liyan Xie, Yao Xie    
*International Conference on Learning Representations* (**ICLR Workshop**) (2021)


- (**DAEL**) 提出了一个统一的DA&DG框架，称为域适应性集合学习：     
[Domain Adaptive Ensemble Learning](https://ieeexplore.ieee.org/abstract/document/9540778)  
Author：Kaiyang Zhou, Yongxin Yang, Yu Qiao, Tao Xiang    
*IEEE Transactions on Image Processing* (**TIP CCF-A**) (2021)  
[[code]](https://github.com/KaiyangZhou/Dassl.pytorch)


- (**DDG**) 通过将语义和变化表征分离到不同的子空间，同时强制执行不变性约束，以学习语义概念的内在表征：  
[Towards Unsupervised Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Towards_Unsupervised_Domain_Generalization_CVPR_2022_paper.html)  
Author：Xingxuan Zhang, Linjun Zhou, Renzhe Xu, Peng Cui, Zheyan Shen, Haoxin Liu    
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)   
[[code]](https://github.com/hlzhang109/DDG)


#### Meta-Learning-Based Methods
> Meta-learning-based methods aim to divide the data form multi-source domains into meta-train and meta-test sets to simulate domain shift.
- (**MetaReg**) 用一个新的正则化函数来编码域泛化的概念，并提出了在 "学会学习"（或）元学习框架中寻找这样一个正则化函数的问题   
[MetaReg: Towards Domain Generalization using Meta-Regularization](https://proceedings.neurips.cc/paper/2018/hash/647bba344396e7c8170902bcf2e15551-Abstract.html)    
Author：Balaji, Yogesh, Swami Sankaranarayanan, and Rama Chellappa   
*Advances in Neural Information Processing Systems* (**NeurIPS**) (2018)


- (**MLDG**) 首次提出用于DG的元学习方法，通过在每个小批次中合成虚拟测试域来模拟训练期间的训练/测试域偏移：  
[Learning to generalize: Meta-learning for domain generalization](https://ojs.aaai.org/index.php/AAAI/article/view/11596)  
Author：Li, Da, Yongxin Yang, Yi-Zhe Song, Timothy M. Hospedales  
*AAAI Conference on Artificial Intelligence* (**AAAI**) 2018  
[[code]](https://github.com/HAHA-DL/MLDG)


- (**MASF**) 采用了一种与模型无关的学习范式，通过基于梯度的元训练和元测试程序，将优化暴露在领域偏移中：  
[Domain Generalization via Model-Agnostic Learning of Semantic Features](https://proceedings.neurips.cc/paper/2019/hash/2974788b53f73e7950e8aa49f3a306db-Abstract.html)  
 Author：Qi Dou, Daniel Coelho de Castro, Konstantinos Kamnitsas, Ben Glocker  
*Advances in Neural Information Processing Systems 32* (**NeurIPS**) (2019)  
[[Code]](https://github.com/biomedia-mira/masf)


- (**M-ADA**) 提出了一种名为对抗性领域增强的新方法来来创建 "虚构 "而又 "具有挑战性 "的样本，进而解决分布外（*OOD*）的泛化问题：  
[Learning to Learn Single Domain Generalization](https://openaccess.thecvf.com/content_CVPR_2020/html/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.html)  
 Author：Fengchun Qiao, Long Zhao, Xi Peng  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2020)  
[[Code]](https://github.com/joffery/M-ADA)

  
- (**Meta-CVAE**) 提出了元条件变异自动编码器（*Meta-CVAE*），一个新的元概率潜变量框架，用于领域泛化：  
[Meta conditional variational auto-encoder for domain generalization](https://reader.elsevier.com/reader/sd/pii/S1077314222000996?token=C03BBC292DFE14380977D9A14DC4A1556884F1DC75F6C20DB4D177C1F88B2DD6C7644F11F7103848C7E0BBA1687CED9E&originRegion=us-east-1&originCreation=20220725025840)  
 Author：Zhiqiang Ge, Zhihuan Song, Xin Li, Lei Zhang  
*Computer Vision and Image Understanding*  (2020)


- (**SBL**) 开发了一种多视图正则化的元学习算法，在更新模型时采用多个任务来产生合适的优化方向。在测试阶段，利用多个增强的图像来产生多视图预测，通过融合测试图像的不同视图的结果来显著提高模型的可靠性：  
[More is Better: A Novel Multi-view Framework for Domain Generalization](https://arxiv.org/abs/2112.12329)  
 Author：Jian Zhang, Lei Qi, Yinghuan Shi, Yang Gao  
*arXiv preprint arXiv:2112.12329* (2021)


- 提出了一种新的元学习方案，该方案具有特征分解能力，它为语义分割推导出具有域泛化保证的域变量特征：  
[Meta-Learned Feature Critics for Domain Generalized Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/9506281)  
 Author：Zu-Yun Shiau, Wei-Wei Lin, Ci-Siang Lin, Yu-Chiang Frank Wang  
*IEEE International Conference on Image Processing* (**ICIP CCF-C**) (2021)


- (**mDSDI**) ：提出了一种mDSDI算法，可以分域特定和域不变特征，并使用元训练方案，以支持特定领域的信息从源域到未见域的适应：  
[Exploiting Domain-Specific Features to Enhance Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/b0f2ad44d26e1a6f244201fe0fd864d1-Abstract.html)  
 Author：Manh-Ha Bui, Toan Tran, Anh Tran, Dinh Phung  
*Neural Information Processing Systems 34* (**NeurIPS**) (2021)   
[[Code]](https://github.com/manhhabui/mDSDI)


- (**M^3L**) 引入了一个元学习策略来模拟训练-测试的过程，还提出了一个元批量规范化层(*MetaBN*)来使元测试特征多样化：   
[Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification](https://openaccess.thecvf.com/content/CVPR2021/html/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.html)  
Author：Yuyang Zhao, Zhun Zhong, Fengxiang Yang, Zhiming Luo, Yaojin Lin, Shaozi Li, Nicu Sebe  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)    
[[Code]](https://github.com/HeliosZhao/M3L)


- (**MetaCNN**) 提出元卷积神经网络，通过将图像的卷积特征分解为元特征，并作为 "视觉词汇"：  
[Meta Convolutional Neural Networks for Single Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Wan_Meta_Convolutional_Neural_Networks_for_Single_Domain_Generalization_CVPR_2022_paper.html)  
 Author：Chaoqun Wan, Xu Shen, Yonggang Zhang, Zhiheng Yin, Xinmei Tian, Feng Gao, Jianqiang Huang, Xian-Sheng Hua  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)


#### Gradient Operation-Based Methods
> Gradient operation-based methods mainly consider using gradient information to force the network learn generalized representations.
- (**MASF**) 采用了一种与模型无关的学习范式，通过基于梯度的元训练和元测试程序，将优化暴露在领域偏移中：  
[Domain Generalization via Model-Agnostic Learning of Semantic Features](https://proceedings.neurips.cc/paper/2019/hash/2974788b53f73e7950e8aa49f3a306db-Abstract.html)  
 Author：Qi Dou, Daniel Coelho de Castro, Konstantinos Kamnitsas, Ben Glocker  
*Advances in Neural Information Processing Systems 32* (**NeurIPS**) (2019)  
[[Code]](https://github.com/biomedia-mira/masf)


- (**DGvGS**) 描述了在领域偏移情况下出现的冲突梯度会降低泛化性能，并设计了基于梯度手术的新型梯度协议策略来减轻其影响：  
[Domain Generalization via Gradient Surgery](https://openaccess.thecvf.com/content/ICCV2021/html/Mansilla_Domain_Generalization_via_Gradient_Surgery_ICCV_2021_paper.html)  
 Author：Lucas Mansilla, Rodrigo Echeveste, Diego H. Milone, Enzo Ferrante  
*International Conference on Computer Vision* (**ICCV**) (2021)  
[[Code]](https://github.com/lucasmansilla/DGvGS)



#### Regularization-Based Methods
- (**MetaReg**) 用一个新的正则化函数来编码域泛化的概念，并提出了在 "学会学习"（或）元学习框架中寻找这样一个正则化函数的问题：   
[MetaReg: Towards Domain Generalization using Meta-Regularization](https://proceedings.neurips.cc/paper/2018/hash/647bba344396e7c8170902bcf2e15551-Abstract.html)    
Author：Balaji, Yogesh, Swami Sankaranarayanan, and Rama Chellappa   
*Advances in Neural Information Processing Systems* (**NeurIPS**) 2018


- (**IRM**) 奠基之作，跳出经验风险最小化--不变风险最小化：   
[Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)   
Author：Arjovsky, Martin and Bottou, Leon and Gulrajani, Ishaan and Lopez-Paz, David     
*arXiv preprint arXiv:1907.02893* (2019)  
[[code]](https://github.com/facebookresearch/InvariantRiskMinimization)


- (**LDDG**) 通过变分编码器学习一个具有代表性的特征空间，并用一个新的线性依赖正则化项来捕捉从不同领域收集的医学数据中的可共享信息，以提升医学图像分类模型泛化能力：  
[Domain Generalization for Medical Imaging Classification with Linear-Dependency Regularization](https://proceedings.neurips.cc/paper/2020/hash/201d7288b4c18a679e48b31c72c30ded-Abstract.html)  
 Author：Haoliang Li, Yufei Wang, Renjie Wan, Shiqi Wang, Tie-Qiang Li, Alex Kot  
*Neural Information Processing Systems 33* (**NeurIPS**) (2020)  
[[Code]](https://github.com/wyf0912/LDDG)


- (**DG_via_ER**) 提出了一个衡量所学特征与类标签之间依赖性的熵正则化项，保证在所有源领域中学习条件不变的特征，从而可以学习具有更好泛化能力的分类器：  
[Domain Generalization via Entropy Regularization](https://proceedings.neurips.cc/paper/2020/hash/b98249b38337c5088bbc660d8f872d6a-Abstract.html)  
 Author：Shanshan Zhao, Mingming Gong, Tongliang Liu, Huan Fu, Dacheng Tao  
*Neural Information Processing Systems 33* (**NeurIPS**) (2020)  
[[Code]](https://github.com/sshan-zhao/DG_via_ER)


- (**RSC**) 引入一种简单的训练启发式方法，以提高跨域泛化能力。这种方法抛弃了与每个周期的高梯度相关的表征，并迫使模型用剩余的信息进行预测：  
[Self-Challenging Improves Cross-Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-030-58536-5_8)  
 Author：Zeyi Huang, Haohan Wang, Eric P. Xing, Dong Huang    
*European Conference on Computer Vision* (**ECCV**) (2020)


- (**SBL**) 开发了一种多视图正则化的元学习算法，在更新模型时采用多个任务来产生合适的优化方向。在测试阶段，利用多个增强的图像来产生多视图预测，通过融合测试图像的不同视图的结果来显著提高模型的可靠性：  
[More is Better: A Novel Multi-view Framework for Domain Generalization](https://arxiv.org/abs/2112.12329)  
 Author：Jian Zhang, Lei Qi, Yinghuan Shi, Yang Gao  
*arXiv preprint arXiv:2112.12329* (2021)

  
- (**MBDG**) 提出了一种具有收敛保证的新型域泛化算法：  
[Model-Based Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/a8f12d9486cbcc2fe0cfc5352011ad35-Abstract.html)  
 Author：Alexander Robey, George J. Pappas, Hamed Hassani  
*Advances in Neural Information Processing Systems 34* (**NeurIPS**) (2021)   
[[Code]](https://github.com/arobey1/mbdg)


- (**SelfReg**) 提出了一种新的基于自监督对比学习的领域泛化的正则化方法，其只使用正面的数据对，解决了由负面数据对采样引起的各种问题：  
[SelfReg: Self-Supervised Contrastive Regularization for Domain Generalization](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_SelfReg_Self-Supervised_Contrastive_Regularization_for_Domain_Generalization_ICCV_2021_paper.html)  
 Author：Daehee Kim, Youngjun Yoo, Seunghyun Park, Jinkyu Kim, Jaekoo Lee  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)


#### Normalization-Based Methods
> Normalization-based methods calibrate data from different domains by normalizing them with their statistic.
- (**DSON**) 采用了多种规范化方法，为各个域的优化设计了归一化层，同时每个领域的学习采用单独的仿生参数：  
[Learning to Optimize Domain Specific Normalization for Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-030-58542-6_5)  
 Author：Seonguk Seo, Yumin Suh, Dongwan Kim, Geeho Kim, Jongwoo Han, Bohyung Han   
*European Conference on Computer Vision* (**ECCV**) (2020)


- (**BNE**) 依靠特定领域的归一化层来分解每个训练领域的独立表征，然后使用这种隐式嵌入来定位来自未知域的未见过的样本：   
[Batch Normalization Embeddings for Deep Domain Generalization](https://arxiv.org/abs/2011.12672)  
Author： Mattia Segu, Alessio Tonioni, Federico Tombari   
*arXiv preprint arXiv:2011.12672* (2020)


- (**SNR**) 提出了一个简单而有效的风格标准化和重构 (*SNR*) 模块,通过归一化 (In-stance Normalization，IN) 过滤掉风格的变化：   
[Style Normalization and Restitution for Generalizable Person Re-Identification](https://openaccess.thecvf.com/content_CVPR_2020/html/Jin_Style_Normalization_and_Restitution_for_Generalizable_Person_Re-Identification_CVPR_2020_paper.html)  
Author： Xin Jin, Cuiling Lan, Wenjun Zeng, Zhibo Chen, Li Zhang   
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2020)


- (**SNR**) 提出了一个简单而有效的风格标准化和重构 (*SNR*) 模块,通过归一化 (In-stance Normalization，IN) 过滤掉风格的变化：   
[Style Normalization and Restitution for Domain Generalization and Adaptation](https://ieeexplore.ieee.org/abstract/document/9513542)  
Author： Xin Jin, Cuiling Lan, Wenjun Zeng, Zhibo Chen   
*IEEE Transactions on Multimedia* (**TMM CCF-B**) (2021)


- (**FACT**) 开发了一种新颖的基于傅里叶的数据增强策略，并引入了一种称为co-teacher regularization的双重形式的一致性损失来学习域不变表征：  
[A Fourier-Based Framework for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.html)  
 Author：Qinwei Xu, Ruipeng Zhang, Ya Zhang, Yanfeng Wang, Qi Tian  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)


- (**GpreBN**) 重新审视了批量归一化(BN)，并提出了一种新的测试阶段的BN层设计：   
[Test-time Batch Normalization](https://arxiv.org/abs/2205.10210)  
 Author：Tao Yang, Shenglong Zhou, Yuwang Wang, Yan Lu, Nanning Zheng  
*arXiv preprint arXiv:2205.10210* (2022)  


#### Causality-Based Methods
> Causality-based methods analyze and address the DG problem from a causal perspective.
- (**IRM**) 奠基之作，跳出经验风险最小化--不变风险最小化：   
[Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)   
Author：Arjovsky, Martin and Bottou, Leon and Gulrajani, Ishaan and Lopez-Paz, David     
*arXiv preprint arXiv:1907.02893* (2019)  
[[code]](https://github.com/facebookresearch/InvariantRiskMinimization)


- (**STEAM**) 揭示了利用域内风格不变性对于提高域泛化的效率也是至关重要的：  
[A Style and Semantic Memory Mechanism for Domain Generalization](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_A_Style_and_Semantic_Memory_Mechanism_for_Domain_Generalization_ICCV_2021_paper.html)  
 Author：Yang Chen, Yu Wang, Yingwei Pan, Ting Yao, Xinmei Tian, Tao Mei  
*International Conference on Computer Vision* (**ICCV**) (2021)


- (**CSG**) 提出了一个基于因果推理的因果语义生成模型 (*CSG*)，以便对语义因素和变化因素进行单独建模：  
[Learning Causal Semantic Representation for Out-of-Distribution Prediction](https://proceedings.neurips.cc/paper/2021/hash/310614fca8fb8e5491295336298c340f-Abstract.html)  
 Author：Chang Liu, Xinwei Sun, Jindong Wang, Haoyue Tang, Tao Li, Tao Qin, Wei Chen, Tie-Yan Liu  
*Neural Information Processing Systems 34* (**NeurIPS**) (2021)   
[[Code]](https://github.com/changliu00/causal-semantic-generative-model)


- (**DRIVE**) 提出了一种基于工具变量的方法来学习条件分布中包含的输入特征和标签之间的领域不变性关系：  
[Learning Domain-Invariant Relationship with Instrumental Variable for Domain Generalization](https://arxiv.org/abs/2110.01438)  
 Author：Junkun Yuan, Xu Ma, Kun Kuang, Ruoxuan Xiong, Mingming Gong, Lanfen Lin  
*arXiv preprint arXiv:2110.01438* (2022)

  
- (**CIRL**) 提出了一般的结构性因果模型,从输入中提取因果因素，然后重构不变的因果机制以提升模型泛化能力：  
[Causality Inspired Representation Learning for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Lv_Causality_Inspired_Representation_Learning_for_Domain_Generalization_CVPR_2022_paper.html)  
 Author：Fangrui Lv, Jian Liang, Shuang Li, Bin Zang, Chi Harold Liu, Ziteng Wang, Di Liu  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)  
[[Code]](https://github.com/BIT-DA/CIRL)


- (**RICE**) 数据生成与因果学习结合，基于修改非因果特征但不改变因果部分的转换，在不明确恢复因果特征的情况下解决OOD问题：  
[Out-of-Distribution Generalization With Causal Invariant Transformations](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Out-of-Distribution_Generalization_With_Causal_Invariant_Transformations_CVPR_2022_paper.html)  
 Author：Ruoyu Wang, Mingyang Yi, Zhitang Chen, Shengyu Zhu  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)  


#### Test-Time-Based Methods
> Test-time-based methods leverage the test data, which is available at test-time, to improve generalization performance without any further model training.
- (**T3A**) 提出了*test-time template adjuster(T3A)*，利用测试数据将调整与预测同时进行：   
[Test-Time Classiﬁer Adjustment Module for Model-Agnostic Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html)  
 Author：Yusuke Iwasawa, Yutaka Matsuo  
*Neural Information Processing Systems 34* (**NeurIPS**) (2021)  
[[Slides]](https://neurips.cc/media/neurips-2021/Slides/27675.pdf)


- (**GpreBN**) 重新审视了批量归一化(BN)，并提出了一种新的测试阶段的BN层设计：   
[Test-time Batch Normalization](https://arxiv.org/abs/2205.10210)  
 Author：Tao Yang, Shenglong Zhou, Yuwang Wang, Yan Lu, Nanning Zheng  
*arXiv preprint arXiv:2205.10210* (2022)  


- (**TASD**) 利用医学分割图像的语义形状先验信息，并设计了具有双一致性的测试阶段适应策略：   
[Single-domain Generalization in Medical Image Segmentation via Test-time Adaptation from Shape Dictionary](https://www.aaai.org/AAAI22Papers/AAAI-852.LiuQ.pdf)  
 Author：Quande Liu, Cheng Chen1, Qi Dou1, Pheng-Ann Heng  
*Association for the Advancement of Artificial Intelligence 36* (**AAAI**) (2022)

### Others
- 将图像通过传统算法转换成shock graph，可以简单理解成一个封闭曲线以及该曲线的中轴线，有了图结构之后再利用GNN来做跨域分类：  
[Shape-Biased Domain Generalization via Shock Graph Embeddings](https://openaccess.thecvf.com/content/ICCV2021/html/Narayanan_Shape-Biased_Domain_Generalization_via_Shock_Graph_Embeddings_ICCV_2021_paper.html)  
 Author：Maruthi Narayanan, Vickram Rajendran, Benjamin Kimia  
*International Conference on Computer Vision* (**ICCV**) (2021) 


- (**VBCLS**) 提出了Deep Frequency Filtering (*DFF*),能够增强可迁移的频率成分，并抑制潜在空间中无益于泛化的成分：  
[Domain Generalization under Conditional and Label Shifts via Variational Bayesian Inference](https://arxiv.org/abs/2107.10931)  
 Author：Xiaofeng Liu, Bo Hu, Linghao Jin, Xu Han, Fangxu Xing, Jinsong Ouyang, Jun Lu, Georges EL Fakhri, Jonghye Woo  
*arXiv preprint arXiv:2107.10931* (2021) 


- (**DFF**) 提出了Deep Frequency Filtering (*DFF*),能够增强可迁移的频率成分，并抑制潜在空间中无益于泛化的成分：  
[Deep Frequency Filtering for Domain Generalization](https://arxiv.org/abs/2203.12198)  
 Author：Shiqi Lin, Zhizheng Zhang, Zhipeng Huang, Yan Lu, Cuiling Lan, Peng Chu, Quanzeng You, Jiang Wang, Zicheng Liu, Amey Parulkar, Viraj Navkal, Zhibo Chen  
*arXiv preprint arXiv:2203.12198* (2022) 


## Single Domain Generalization
> Single domain generalization aims learn a generalized model only use one domain, which is more challenge but more realistic.
- (**GUD**) 一种新的对抗性数据增强方法用于解决单一源域泛化问题，该方法可以在未见过的数据分布中学习到更好的泛化性：  
[Generalizing to Unseen Domains via Adversarial Data Augmentation](https://proceedings.neurips.cc/paper/2018/hash/1d94108e907bb8311d8802b48fd54b4a-Abstract.html)  
 Author：Riccardo Volpi, Hongseok Namkoong, Ozan Sener, John C. Duchi, Vittorio Murino, Silvio Savarese  
*Advances in Neural Information Processing Systems 31* (**NeurIPS**) (2018)  
[[Code]](https://github.com/ricvolpi/generalize-unseen-domains)


- (**M-ADA**) 提出了一种名为对抗性领域增强的新方法来来创建 "虚构 "而又 "具有挑战性 "的样本，进而解决分布外（*OOD*）的泛化问题：  
[Learning to Learn Single Domain Generalization](https://openaccess.thecvf.com/content_CVPR_2020/html/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.html)  
 Author：Fengchun Qiao, Long Zhao, Xi Peng  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2020)  
[[Code]](https://github.com/joffery/M-ADA)


- (**L2D**) 提出了一个风格互补模块，通过合成与源域互补的不同分布的图像来提高模型的泛化能力：  
[Learning To Diversify for Single Domain Generalization](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Learning_To_Diversify_for_Single_Domain_Generalization_ICCV_2021_paper.html)  
 Author：Zijian Wang, Yadan Luo, Ruihong Qiu, Zi Huang, Mahsa Baktashmotlagh  
*International Conference on Computer Vision* (**ICCV**) (2021)   
[[Code]](https://github.com/BUserName/Learning_to_diversify)


- (**ASR**) 在对抗性领域增强(*ADA*)过程中，提出了一种通用的归一化方法adaptive standardization and rescaling normalization (*ASR-Norm*)：  
[Adversarially Adaptive Normalization for Single Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Fan_Adversarially_Adaptive_Normalization_for_Single_Domain_Generalization_CVPR_2021_paper.html)  
Author：Xinjie Fan, Qifei Wang, Junjie Ke, Feng Yang, Boqing Gong, Mingyuan Zhou  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)


- (**UMGUD**) 通过于单一源域中获取泛化的不确定性，并利用它来指导输入和标签的扩增，以实现强大的泛化：  
[Adversarially Adaptive Normalization for Single Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.html?ref=https://githubhelp.com)  
Author：Fengchun Qiao, Xi Peng  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)  
[[Code]](https://github.com/joffery/UMGUD)


- (**PDEN**) 提出了一个新颖的渐进式域扩展网络 (*PDEN*) 学习框架，通过逐渐生成模拟目标与数据，提升模型泛化能力：   
[Progressive Domain Expansion Network for Single Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.html)  
Author：Fengchun Qiao, Xi Peng  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)    
[[Code]](https://github.com/lileicv/PDEN)


- (**ACVC**) 通过量化来自单一来源的泛化不确定性，并利用它来指导特征和标签的增广，以实现强大的泛化：  
[Out-of-Domain Generalization From a Single Source: An Uncertainty Quantification Approach](https://ieeexplore.ieee.org/abstract/document/9801711)  
 Author：Xi Peng, Fengchun Qiao, Long Zhao  
*IEEE Transactions on Pattern Analysis and Machine Intelligence* (**TPAMI**) (2022 CCFA)


- (**TASD**) 利用医学分割图像的语义形状先验信息，并设计了具有双一致性的测试阶段适应策略，用于解决医学图像分割的单一源域的泛化问题：  
[Single-domain Generalization in Medical Image Segmentation via Test-time Adaptation from Shape Dictionary](https://www.aaai.org/AAAI22Papers/AAAI-852.LiuQ.pdf)  
 Author：Quande Liu, Cheng Chen1, Qi Dou1, Pheng-Ann Heng  
*Association for the Advancement of Artificial Intelligence 36* (**AAAI**) (2022)


- (**TASD**) 利用医学分割图像的语义形状先验信息，并设计了具有双一致性的测试阶段适应策略：   
[Single-domain Generalization in Medical Image Segmentation via Test-time Adaptation from Shape Dictionary](https://www.aaai.org/AAAI22Papers/AAAI-852.LiuQ.pdf)  
 Author：Quande Liu, Cheng Chen1, Qi Dou1, Pheng-Ann Heng  
*Association for the Advancement of Artificial Intelligence 36* (**AAAI**) (2022)


- (**MetaCNN**) 提出元卷积神经网络，通过将图像的卷积特征分解为元特征，并作为 "视觉词汇"：  
[Meta Convolutional Neural Networks for Single Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Wan_Meta_Convolutional_Neural_Networks_for_Single_Domain_Generalization_CVPR_2022_paper.html)  
 Author：Chaoqun Wan, Xu Shen, Yonggang Zhang, Zhiheng Yin, Xinmei Tian, Feng Gao, Jianqiang Huang, Xian-Sheng Hua  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)


## Self-Supervised Domain Generalization
> Self-supervised domain generalization methods improve model generalization ability by solving some pretext tasks with data itself.
- (**JiGen**) 以监督的方式学习语义标签，并通过从自我监督的信号中学习如何解决相同图像上的拼图来提升泛化能力：  
[Domain Generalization by Solving Jigsaw Puzzles](https://openaccess.thecvf.com/content_CVPR_2019/html/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.html)  
 Author：Fabio M. Carlucci, Antonio D'Innocente, Silvia Bucci, Barbara Caputo, Tatiana Tommasi  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2019)  
[[Code]](https://github.com/fmcarlucci/JigenDG)


- (**PDEN**) 提出了一个新颖的渐进式域扩展网络 (*PDEN*) 学习框架，通过逐渐生成模拟目标与数据，提升模型泛化能力：   
[Progressive Domain Expansion Network for Single Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.html)  
Author：Fengchun Qiao, Xi Peng  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)    
[[Code]](https://github.com/lileicv/PDEN)


- (**EISNet**) 提出了一个新的领域泛化框架（称为EISNet），利用多任务学习范式，从多源领域的图像的外在关系监督和内在自我监督中同时学习如何跨领域泛化：  
[Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization](https://link.springer.com/chapter/10.1007/978-3-030-58545-7_10)  
 Author：Shujun Wang, Lequan Yu, Caizi Li, Chi-Wing Fu, Pheng-Ann Heng   
*Proceedings of the European Conference on Computer Vision* (**ECCV**) 2020  
[[code]](https://github.com/EmmaW8/EISNet)


- (**ATSRL**) 多视角学习，提出对抗性师生表征学习框架，将表征学习和数据增广相结合，前者逐步更新教师网络以得出域通用的表征，而后者则合成数据的外源但合理的分布：  
[Adversarial Teacher-Student Representation Learning for Domain Generalization](https://proceedings.neurips.cc/paper/2021/hash/a2137a2ae8e39b5002a3f8909ecb88fe-Abstract.html)  
 Author：Fu-En Yang, Yuan-Chia Cheng, Zu-Yun Shiau, Yu-Chiang Frank Wang  
*Advances in Neural Information Processing Systems 34* (**NeurIPS**) (2021)  


- (**SelfReg**) 提出了一种新的基于自监督对比学习的领域泛化的正则化方法，其只使用正面的数据对，解决了由负面数据对采样引起的各种问题：  
[SelfReg: Self-Supervised Contrastive Regularization for Domain Generalization](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_SelfReg_Self-Supervised_Contrastive_Regularization_for_Domain_Generalization_ICCV_2021_paper.html)  
 Author：Daehee Kim, Youngjun Yoo, Seunghyun Park, Jinkyu Kim, Jaekoo Lee  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)


## Semi/Weak/Un-Supervised Domain Generalization
> Semi/weak-supervised domain generalization assumes that a part of the source data is unlabeled, while unsupervised domain generalization assumes no training supervision.
- (**DSBF**) 研究了只有一个源域被标记的单标记域泛化（*SLDG*）任务,提出了一种 Domain-Specific Bias Filtering (*DSBF*)的方法，用标记的源数据初始化一个判别模型，然后用未标记的源数据过滤掉其领域特定偏见，以提高泛化能力：  
[Domain-Specific Bias Filtering for Single Labeled Domain Generalization](https://arxiv.org/abs/2110.00726)  
Author：Junkun Yuan, Xu Ma, Defang Chen, Kun Kuang, Fei Wu, Lanfen Lin    
*arXiv preprint arXiv:2110.00726* (2021)    
[[Code]](https://github.com/junkunyuan/DSBF)


- (**semanticGAN**) 训练生成式对抗网络以捕捉图像-标签的联合分布，并使用大量的未标记图像和少量的标记图像进行有效的训练：   
[Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Semantic_Segmentation_With_Generative_Models_Semi-Supervised_Learning_and_Strong_Out-of-Domain_CVPR_2021_paper.html)  
Author：Daiqing Li, Junlin Yang, Karsten Kreis, Antonio Torralba, Sanja Fidler  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)


- (**SSDG**) 利用多源、部分标签的训练数据学习一个可泛化的领域模型：  
[Semi-Supervised Domain Generalization with Stochastic StyleMatch](https://arxiv.org/abs/2106.00592)  
Author：Kaiyang Zhou, Chen Change Loy, Ziwei Liu    
*arXiv preprint arXiv:2106.00592* (2021)    
[[Code]](https://github.com/KaiyangZhou/ssdg-benchmark)


- (**FlexMatch**) 提出半监督学习法FlexMatch和开源库TorchSSL：   
[FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](https://proceedings.neurips.cc/paper/2021/hash/995693c15f439e3d189b06e89d145dd5-Abstract.html)  
 Author：Bowen Zhang, Yidong Wang, Wenxin Hou, HAO WU, Jindong Wang, Manabu Okumura, Takahiro Shinozaki    
*Neural Information Processing Systems 34* (**NeurIPS**) (2021)  
[[Slides]](https://github.com/TorchSSL/TorchSSL)  [[Video]](https://www.zhihu.com/zvideo/1441001725339987968)


- (**BrAD**) 提出了一种新颖的自监督跨域学习方法，将所有的域在语义上与一个共同的辅助桥域进行对齐：  
[Unsupervised Domain Generalization by Learning a Bridge Across Domains](https://openaccess.thecvf.com/content/CVPR2022/html/Harary_Unsupervised_Domain_Generalization_by_Learning_a_Bridge_Across_Domains_CVPR_2022_paper.html)  
Author：Sivan Harary, Eli Schwartz, Assaf Arbelle, Peter Staar, Shady Abu-Hussein, Elad Amrani, Roei Herzig, Amit Alfassy, Raja Giryes, Hilde Kuehne, Dina Katabi, Kate Saenko, Rogerio S. Feris, Leonid Karlinsky    
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)    


- (**DARLING**) 关注模型预训练的过程对DG任务的影响，设计了一个在DG数据集无监督预训练的算法：  
[Towards Unsupervised Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Towards_Unsupervised_Domain_Generalization_CVPR_2022_paper.html)  
Author：Xingxuan Zhang, Linjun Zhou, Renzhe Xu, Peng Cui, Zheyan Shen, Haoxin Liu    
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)  


- (**PCL**) 提出了一种基于代理的对比学习方法，用代理对样本的关系取代了原来对比学习中的样本对样本的关系，缓解了正向对齐问题：  
[PCL: Proxy-Based Contrastive Learning for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Yao_PCL_Proxy-Based_Contrastive_Learning_for_Domain_Generalization_CVPR_2022_paper.html)  
Author：Xufeng Yao, Yang Bai, Xinyun Zhang, Yuechen Zhang, Qi Sun, Ran Chen, Ruiyu Li, Bei Yu    
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2022)  


## Open/Heterogeneous Domain Generalization
> Open/heterogeneous domain generalization assumes the label space of one domain is different from that of another domain.
- 考虑了更具挑战性的异质领域泛化设置，即未见过的领域与已见过的领域不共享标签空间，目标是训练一个对新数据和新类别有用的现成的特征表示：    
[Feature-Critic Networks for Heterogeneous Domain Generalization](https://proceedings.mlr.press/v97/li19l.html)       
Author：Yiying Li, Yongxin Yang, Wei Zhou, Timothy Hospedales      
*International Conference on Machine Learning* (**PMLR-ICML**) (2019)   
[[Code](https://github.com/liyiying/Feature_Critic)]


- 提出了一种新的异质域泛化方法，即用两种不同的采样策略将多个源域的样本混合起来：  
[Heterogeneous Domain Generalization Via Domain Mixup](https://ieeexplore.ieee.org/abstract/document/9053273)     
Author：Yufei Wang, Haoliang Li, Alex C. Kot    
*IEEE International Conference on Acoustics, Speech and Signal Processing* (**ICASSP CCF-B**) (2020)   
[[Code](https://github.com/wyf0912/MIXALL)]


- (**SnMpNet**) 首次解决了通用跨域检索的问题，其中测试数据可能属于训练期间未见过的类或域：  
[Universal Cross-Domain Retrieval: Generalizing Across Classes and Domains](https://openaccess.thecvf.com/content/ICCV2021/html/Paul_Universal_Cross-Domain_Retrieval_Generalizing_Across_Classes_and_Domains_ICCV_2021_paper.html)  
Author：Soumava Paul, Titir Dutta, Soma Biswas    
*International Conference on Computer Vision* (**ICCV**) (2021)  

## Federated Domain Generalization
> Federated domain generalization assumes that source data is non-shared and can not be collected to train a centralized model for data privacy and transmission restrictions.
- (**ELCFS**) 提出了Episodic Learning in Continuous Frequency Space (*ELCFS*)方法，使每个客户端能够在数据分散的挑战性约束下利用多源数据的分布，解决联邦场景下的分割域泛化任务：   
[FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.html)  
Author：Quande Liu, Cheng Chen, Jing Qin, Qi Dou, Pheng-Ann Heng  
*Conference on Computer Vision and Pattern Recognition* (**CVPR**) (2021)    
[[Code]](https://github.com/liuquande/FedDG-ELCFS)

  
- (**COPA**) 提出了一种叫做Collaborative Optimization and Aggregation(*COPA*)的联邦学习框架，为分散的DG和UDA优化一个广义的目标模型：   
[Collaborative Optimization and Aggregation for Decentralized Domain Generalization and Adaptation](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_Collaborative_Optimization_and_Aggregation_for_Decentralized_Domain_Generalization_and_Adaptation_ICCV_2021_paper.html)  
Author：Guile Wu, Shaogang Gong  
*International Conference on Computer Vision* (**PMLR-ICML**) (2021)    


- (**FedADG**) 提出了FedADG，采用了联邦对抗学习的方法，通过将每个分布与参考分布相匹配来衡量和调整不同源域之间的分布：   
[Federated Learning with Domain Generalization](https://arxiv.org/abs/2111.10487)  
Author：Liling Zhang, Xinyu Lei, Yichun Shi, Hongyu Huang, Chao Chen   
*arXiv preprint arXiv:2111.10487* (2021)    
[[Code]](https://github.com/wzml/FedADG)


- (**CSAC**) 提出了协作语义聚合和校准(*CSAC*)，通过充分吸收多源语义信息，同时避免不安全的数据融合来解决联邦场景下的领域泛化任务：   
[Collaborative Semantic Aggregation and Calibration for Separated Domain Generalization](https://ui.adsabs.harvard.edu/abs/2021arXiv211006736Y/abstract)  
Author：Junkun Yuan, Xu Ma, Defang Chen, Kun Kuang, Fei Wu, Lanfen Lin   
*arXiv preprint arXiv:2110.06736* (2021)    
[[Code]](https://github.com/junkunyuan/CSAC)


# Datasets
> We list the widely used benchmark datasets for domain generalization including classification and segmentation. 

|                                                                 Dataset                                                                  | Task                       | #Domain | #Class | #Sample |                                                                 Description                                                                 |
|:----------------------------------------------------------------------------------------------------------------------------------------:|----------------------------|:-------:|:------:|:-------:|:-------------------------------------------------------------------------------------------------------------------------------------------:|
|                                 [PACS](https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd)                                 | Classification             |    4    |   7    |  9,991  |                                                         Art, Cartoon, Photo, Sketch                                                         |
|                                 [VLCS](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8)                                 | Classification             |    4    |   5    | 10,729  |                                                     Caltech101, LabelMe, SUN09, VOC2007                                                     |
|                             [Office-Home](https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC)                              | Classification             |    4    |   65   | 15,588  |                                                         Art, Clipart, Product, Real                                                         |
|                          [Office-31](https://mega.nz/file/dSpjyCwR#9ctB4q1RIE65a4NoJy0ox3gngh15cJqKq1XpOILJt9s)                          | Classification             |    3    |   31   |  4,110  |                                                            Amazon, Webcam, DSLR                                                             |
|                         [Office-Caltech](https://drive.google.com/file/d/14OIlzWFmi5455AjeBZLak2Ku-cFUrfEo/view)                         | Classification             |    4    |   10   |  2,533  |                                                        Caltech, Amazon, Webcam, DSLR                                                        |
|                           [Digits-DG](https://drive.google.com/file/d/1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm/view)                            | Classification             |    4    |   10   | 24,000  |                                                          MNIST, MNIST-M, SVHN, SYN                                                          |
|                      [Digit-5](https://drive.google.com/file/d/15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7/view?usp=sharing)                       | Classification             |    5    |   10   | ~10,000 |                                                       MNIST, MNIST-M, SVHN, SYN, USPS                                                       |
|                                            [Rotated MNIST](https://github.com/Emma0118/mate)                                             | Classification             |    6    |   10   |  7,000  |                                                   Rotated degree: {0, 15, 30, 45, 60, 75}                                                   |
|                                            [Colored MNIST](https://github.com/Emma0118/mate)                                             | Classification             |    3    |   2    |  7,000  |                                                       Colored degree: {0.1, 0.3, 0.9}                                                       |
|                                       [CIFAR-10-C](https://zenodo.org/record/2535967#.YuD3ly-KGu4)                                       | Classification             |   --    |   4    | 60,000  |   The test data are damaged by 15 corruptions (each with 5 intensity levels) drawn from 4 categories (noise, blur, weather, and digital)    |
|                                      [CIFAR-100-C](https://zenodo.org/record/3555552#.YuD31C-KGu4)                                       | Classification             |   --    |   4    | 60,000  |   The test data are damaged by 15 corruptions (each with 5 intensity levels) drawn from 4 categories (noise, blur, weather, and digital)    |
|                                                   [DomainNet](http://ai.bu.edu/M3SDA/)                                                   | Classification             |    6    |  345   | 586,575 |                                           Clipart, Infograph, Painting, Quick-draw, Real, Sketch                                            |
|                                            [miniDomainNet](https://arxiv.org/abs/2003.07325)                                             | Classification             |    4    |  345   | 140,006 |                           A smaller and less noisy version of DomainNet including Clipart, Painting, Real, Sketch                           |
|                                  [VisDA-17](https://github.com/VisionLearningGroup/taskcv-2017-public)                                   | Classification             |    3    |   12   | 280,157 |                                                3 domains of synthetic-to-real generalization                                                |
|               [Terra Incognita](https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz)               | Classification             |    4    |   10   | 24,788  |                                          Wild animal images taken at locations L100, L38, L43, L46                                          |
|                                            [Prostate MRI](https://liuquande.github.io/SAML/)                                             | Medical image segmentation |    6    |   --   |   116   |                                    Contains prostate T2-weighted MRI data from 6 institutions: Site A~F                                     |
|                          [Fundus OC/OD](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view)                          | Medical image segmentation |    4    |   --   |  1060   |                                            Contains fundus images from 4 institutions: Site A~D                                             |
| [GTA5-Cityscapes]([GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) and [Cityscapes](https://www.cityscapes-dataset.com)) | Semantic segmentation      |    2    |   --   | 29,966  |                                                2 domains of synthetic-to-real generalization                                                |


# Libraries
> We list the libraries of domain generalization.
- [Transfer Learning Library (thuml)](https://github.com/thuml/Transfer-Learning-Library) for Domain Adaptation, Task Adaptation, and Domain Generalization.
- [DomainBed (facebookresearch)](https://github.com/facebookresearch/DomainBed)  is a suite to test domain generalization algorithms.
- [DeepDG (Jindong Wang)](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG): Deep domain generalization toolkit, which is easier then DomainBed.
- [Dassl (Kaiyang Zhou)](https://github.com/KaiyangZhou/Dassl.pytorch): A PyTorch toolbox for domain adaptation, domain generalization, and semi-supervised learning.
- [TorchSSL (Jindong Wang)](https://github.com/TorchSSL/TorchSSL): A open library for semi-supervised learning.

# Other Resources
- A collection of domain generalization papers organized by  [amber0309](https://github.com/amber0309/Domain-generalization).
- A collection of domain generalization papers organized by [jindongwang](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#domain-generalization).
- A collection of papers on domain generalization, domain adaptation, causality, robustness, prompt, optimization, generative model, etc, organized by [yfzhang114](https://github.com/yfzhang114/Generalization-Causality).
- A collection of awesome things about domain generalization organized by [junkunyuan](https://github.com/junkunyuan/Awesome-Domain-Generalization).

# Contact
- If you would like to add/update the latest publications / datasets / libraries, please directly add them to this `README.md`.
- If you would like to correct mistakes/provide advice, please contact us by email (nzw@zju.edu.cn).
- You are welcomed to update anything helpful.

# Acknowledgements
- We refer to [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://ieeexplore.ieee.org/abstract/document/9782500) to design the hierarchy of the [Contents](#contents).
- We refer to [junkunyuan](https://github.com/junkunyuan/Awesome-Domain-Generalization), [amber0309](https://github.com/amber0309/Domain-generalization), and [yfzhang114](https://github.com/yfzhang114/Generalization-Causality) to design the details of the papers and datasets.
