# Accelerating Materials Discovery through Active Learning: Methods, Challenges and Opportunities
## Abstract
The convergence of large-scale experimentation and computation has transformed materials science into a discipline where data plays a central role. However, the vast design space and high costs of experiments and simulations still slow progress. In contrast to traditional approaches, active learning (AL) identifies which experiments or simulations will provide the most valuable information. This minimizes unnecessary work and cost. By prioritizing the most informative data points, AL enables more efficient and targeted exploration, particularly when resources are limited. This approach accelerates breakthroughs in complex systems, such as superconductors, catalysts, and batteries. This review introduces a framework that categorizes AL in both experimental and computational settings. It highlights the importance of integrating domain knowledge and addresses interpretability challenges. By providing a pragmatic roadmap, it makes a strong case for adopting AL, especially to maximize impact in resource-constrained materials research.
## Key Themes
* **Core Problem**:Materials discovery faces exponential combinatorial complexity (e.g., multi-component alloys) and multi-scale physics, making trial-and-error inefficient.
* **AL Solution**:AL prioritizes "informative" samples for labeling, forming closed-loop workflows with surrogate models, query strategies, and experimental/computational oracles.
* **Applications**: Compositional selection, structural design, processing optimization, and property prediction.
* **Innovations**:Unified mathematical formulation for pool-based and generative-based AL; emphasis on knowledge-integrated methods.
## Introduction
### Introduction to AL
Supervised learning problems are plagued by the high cost of labeling and the difficulty of obtaining large quantities of labels. For certain tasks, only domain experts can accurately label samples. In this context, active learning (AL) attempts to train high-performing models by selectively labeling less data.
The key assumption of active learning is that different samples have varying degrees of importance for a given task, and therefore, the performance gains they bring are not uniform. Selecting more important samples allows the current model to achieve better performance with fewer labeled samples. In this process, the essence of active learning is to evaluate the importance of samples (e.g., their information content, expected performance, etc.). Most research focuses on how to evaluate samples.

However, as the field evolves and the literature expands, the term active learning may have different connotations. Generally speaking, when we talk about active learning, we mean:

* From a problem perspective: a machine learning approach that reduces labeling costs by using some active strategy to construct a smaller training set.
* From a strategy perspective: an assessment of the importance of unlabeled samples in some way.
* From a training perspective: an interactive labeling, training, and evaluation process.
### Why Crucial for Materials Science?
* Experiments (e.g., synthesis) or simulations (e.g., DFT) are expensive (hours/days, high reagents/manpower).
* Design spaces are vast: e.g., $10^{60}$ possible high-entropy alloys from 5+ elements.
* AL reduces queries by 50-90% (e.g., mapping performance landscapes with 10s vs. 100s of experiments).

## Application of AL in the field of materials
We introduces a task-oriented classification of Active Learning (AL) methods, emphasizing their diversity in reducing labeling costs and speeding up discovery. It organizes AL along two orthogonal axes: acquisition paradigms (pool-based vs. generative-based) and knowledge integration (purely data-driven vs. domain-informed). This framework provides a conceptual foundation, distinguishing what to optimize (objectives) from how candidates are sourced (acquisition), and unifies variants under an iterative acquisition-update process tailored to materials science constraints.
### Pool-Based Active Learning for Materials Science
Pool-based AL, the most widely adopted paradigm in the field of materials discovery, centers on selecting the most valuable samples for labeling from a static candidate materials pool.The effectiveness of pool-based methods relies heavily on whether the selected model is well-suited to the current task’s data structure, feature modalities, and prediction objectives. Different types of learning models exhibit significant variations in uncertainty quantification, sampling strategies, and feature representation capabilities when handling diverse materials data types. The following discussion explores several primary model types used in pool-based AL for materials discovery and their corresponding application strategies, focusing on model-task compatibility.

| Model Types | Primary material task types | Representative references |
| :-----| :----- | :----- |
| Probabilistic and Bayesian Models | Composition screening; property prediction; multi-objective optimization | Composition screening:24 27 28 29 30 31 34 35 36 38 39 32; Process condition optimization / synthesis optimization:26 |
| Neural Networks | High-dimensional property prediction; surrogate models for potential-energy surfaces; Crystal/molecular structure designand discovery; image/spectral characterization | Structure design and discovery:41 23 25 42 43 46 48 47 50 51; Composition screening:49 52} |

### Generative Active Learning for Materials Science
Generative AL expands the exploration boundaries of materials discovery by integrating candidate sample generation with uncertainty assessment. Unlike pool-based methods that rely on pre-constructed static sample libraries, the generative paradigm utilizes generative models to ”instantly” construct efficient candidates within the materials design space, thereby transcending the limitations of existing samples in extremely high-dimensional combinatorial spaces. On one hand, generative models learn the latent distribution between material attributes and strucures, enabling the synthesis of ”novel” chemical compositions or microstructures with potential performance advantages. On the other hand, incorporating uncertainty quantification methods into the generative loop allows for the prioritized generation and labeling of samples expected  to provide the highest information gain for the surrogate model in each iteration, maximizing the utilization efficiency of experimental/computational resources.
|Generative Model Type | Primary material task types | Representative references |
| :-----| :----- | :----- |
| Evolutionary Algorithm-Based Generative AL | Combinatorial/discrete design exploration; multi-objective optimization | Structure design and discovery:56 57  |
| VAE-Based Generative AL | Conditional/target-oriented generation; latent-space optimization; virtual library expansion | Composition screening:21 58 59 60 61 |
| GAN-Based Generative AL | High-diversity candidate generation; exploratory discovery; few-shot target search | Composition screening:61 62 63,64; Structure design and discovery:65 |
| LLMs-Based AL | Textual knowledge extraction; generative proposal of compositions/structures/recipes; human-machine collaboration | Structure design and discovery:67; Process condition optimization / synthesis optimization:66,68,69 |

## References

[1. Batra, R., Song, L., and Ramprasad, R. (2021). Emerging materials intelligence ecosystems propelled by machine learning. Nature Reviews Materials, 6, 655–678.](https://www.nature.com/articles/s41578-020-00255-y)

[2. Gubernatis, J., and Lookman, T. (2018). Machine learning in materials design and discovery: Examples from the present and suggestions for the future. Physical Review Materials, 2, 120301.](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.2.120301)

[3. Liu, Y., Zhao, T., Ju, W., and Shi, S. (2017). Materials discovery and design using machine learning. Journal of Materiomics, 3, 159–177.](https://www.sciencedirect.com/science/article/pii/S2352847817300515)

[4. Zheng, Y., Xu, H., Li, Z., Li, L., Yu, Y., Jiang, P., Shi, Y., Zhang, J., Huang, Y., Luo, Q. et al.(2025). Artificial intelligence-driven approaches in semiconductor research. Advanced Materials pp. 2504378](https://pubmed.ncbi.nlm.nih.gov/40534303/)

[5. Slattery, A., Wen, Z., Tenblad, P., Sanjosé-Orduna, J., Pintossi, D., den Hartog, T., and Noël, T. (2024). Automated self-optimization, intensification, and scale-up of photocatalysis in flow. Science, 383, eadj1817.](https://www.science.org/doi/10.1126/science.adj1817)

[6. Lookman, T., Balachandran, P.V., Xue, D., and Yuan, R. (2019). Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design. npj Computational Materials, 5, 21.](https://www.nature.com/articles/s41524-019-0153-8)

[7.  Bai, X., and Zhang, X. (2025). Artificial intelligence-powered materials science. Nano-Micro Letters, 17, 1–30.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11803041/)

[8. Shahriari, B., Swersky, K., Wang, Z., Adams, R.P., and De Freitas, N. (2015). Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE, 104, 148–175.](https://ieeexplore.ieee.org/document/7352306)

[9.Szymanski, N.J., Rendy, B., Fei, Y., Kumar, R.E., He, T., Milsted, D., McDermott, M.J., Gallant, M., Cubuk, E.D., Merchant, A., et al. (2023). An autonomous laboratory for the accelerated synthesis of novel materials. Nature, 624, 86–91.](https://www.nature.com/articles/s41586-023-06734-w)

[10. MacLeod, B.P., Parlane, F.G., Morrissey, T.D., Häse, F., Roch, L.M., Dettelbach, K.E., Moreira, R., Yunker, L.P., Rooney, M.B., Deeth, J.R., et al. (2020). Self-driving laboratory for accelerated discovery of thin-film materials. Science Advances, 6, eaaz8867.](https://www.science.org/doi/10.1126/sciadv.aaz8867)

[11. Settles, B. (1995). Active learning literature survey. Science 10, 237–304.](https://burrsettles.com/pub/settles.activelearning.pdf)

[12. Schmidt, J., Marques, M.R., Botti, S., and Marques, M.A. (2019). Recent advances and applications of machine learning in solid-state materials science. npj Computational Materials, 5, 83.](https://www.nature.com/articles/s41524-019-0221-0)

[13. Kumar, P., and Gupta, A. (2020). Active learning query strategies for classification, regression, and clustering: A survey. Journal of Computer Science and Technology, 35, 913–945.](https://link.springer.com/article/10.1007/s11390-020-9487-4)

[14. Saal, J.E., Oliynyk, A.O., and Meredig, B. (2020). Machine learning in materials discovery: Confirmed predictions and their underlying approaches. Annual Review of Materials Research, 50, 49–69.](https://www.annualreviews.org/content/journals/10.1146/annurev-matsci-090319-010954)

[15. Yunfan, W., Yuan, T., Yumei, Z., and Dezhen, X. (2023). Progress on active learning assisted materials discovery. Journal of the Chinese Ceramic Society, 51, 544–551.](https://www.sciopen.com/article/10.14062/j.issn.0454-5648.20220924)

[16. Lewis, D.D. (1995). A sequential algorithm for training text classifiers: Corrigendum and additional data. ACM SIGIR Forum, 29, 13–19.](https://dl.acm.org/doi/10.1145/219587.219592)

[17. Beluch, W.H., Genewein, T., Nürnberger, A., and Köhler, J.M. (2018). The power of ensembles for active learning in image classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 9368–9377.](https://ieeexplore.ieee.org/document/8579074)

[18. Zhu, M., Fan, C., Chen, H., Liu, Y., Mao, W., Xu, X., and Shen, C. (2024). Generative active learning for long-tailed instance segmentation. In International Conference on Machine Learning, PMLR, 62349–62368.](https://openreview.net/forum?id=Pt7WaT5dxV)

[19. Zhu, J.J., and Bento, J. (2017). Generative adversarial active learning. arXiv preprint arXiv:1702.07956.](https://ui.adsabs.harvard.edu/abs/2017arXiv170207956Z/abstract)

[20. Liu, Y., Li, Z., Zhou, C., Jiang, Y., Sun, J., Wang, M., and He, X. (2019). Generative adversarial active learning for unsupervised outlier detection. IEEE Transactions on Knowledge and Data Engineering, 32, 1517–1528.](https://arxiv.org/abs/1702.07956)

[21. Gui, J., Sun, Z., Wen, Y., Tao, D., and Ye, J. (2021). A review on generative adversarial networks: Algorithms, theory, and applications. IEEE Transactions on Knowledge and Data Engineering, 35, 3313–3332.](https://ieeexplore.ieee.org/document/9625798)

[22. Rao, Z., Tung, P.Y., Xie, R., Wei, Y., Zhang, H., Ferrari, A., Klaver, T., Körmann, F., Sukumar, P.T., Kwiatkowski da Silva, A., et al. (2022). Machine learning–enabled high-entropy alloy discovery. Science, 378, 78–85.](https://www.science.org/doi/10.1126/science.abo4940)

[23. Sohail, Y., Zhang, C., Xue, D., Zhang, J., Zhang, D., Gao, S., Yang, Y., Fan, X., Zhang, H., Liu, G., et al. (2025). Machine-learning design of ductile FeNiCoAlTa alloys with high strength. Nature, 1–6.](https://www.nature.com/articles/s41586-025-09160-2)

[24. Moon, J., Beker, W., Siek, M., Kim, J., Lee, H.S., Hyeon, T., and Grzybowski, B.A. (2024). Active learning guides discovery of a champion four-metal perovskite oxide for oxygen evolution electrocatalysis. Nature Materials, 23, 108–115.](https://www.nature.com/articles/s41563-023-01707-w)

[25. Suvarna, M., Zou, T., Chong, S.H., Ge, Y., Martín, A.J., and Pérez-Ramírez, J. (2024). Active learning streamlines development of high performance catalysts for higher alcohol synthesis. Nature Communications, 15, 5844.](https://www.nature.com/articles/s41467-024-50215-1)

[26. Merchant, A., Batzner, S., Schoenholz, S.S., Aykol, M., Cheon, G., and Cubuk, E.D. (2023). Scaling deep learning for materials discovery. Nature, 624, 80–85.](https://www.nature.com/articles/s41586-023-06735-9)

[27. Lee, J.A., Park, J., Sagong, M.J., Ahn, S.Y., Cho, J.W., Lee, S., and Kim, H.S. (2025). Active learning framework to optimize process parameters for additive-manufactured Ti-6Al-4V with high strength and ductility. Nature Communications, 16, 931.](https://www.nature.com/articles/s41467-025-56267-1)

[28. Zhang, R., Xu, J., Zhang, H., Xu, G., and Luo, T. (2025). Active learning-guided exploration of thermally conductive polymers under strain. Digital Discovery, 4, 812–823.](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00267a)

[29. Johnson, N.S., Mishra, A.A., Kirsch, D.J., and Mehta, A. (2024). Active learning for rapid targeted synthesis of compositionally complex alloys. Materials, 17, 4038.](https://www.mdpi.com/1996-1944/17/16/4038)

[30. Verduzco, J.C., Marinero, E.E., and Strachan, A. (2021). An active learning approach for the design of doped LLZO ceramic garnets for battery applications. Integrating Materials and Manufacturing Innovation, 10, 299–310.](https://link.springer.com/article/10.1007/s40192-021-00214-7)

[31. Pedersen, J.K., Clausen, C.M., Krysiak, O.A., Xiao, B., Batchelor, T.A., Löffler, T., Mints, V.A., Banko, L., Arenz, M., Savan, A., et al. (2021). Bayesian optimization of high-entropy alloy compositions for electrocatalytic oxygen reduction. Angewandte Chemie, 133, 24346–24354.](https://onlinelibrary.wiley.com/doi/full/10.1002/anie.202108116)

[32. Terayama, K., Tamura, R., Nose, Y., Hiramatsu, H., Hosono, H., Okuno, Y., and Tsuda,K. (2019). Efficient construction method for phase diagrams using uncertainty sampling.Physical Review Materials 3, 033802. ](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.3.033802)

[33. Talapatra, A., Boluki, S., Duong, T., Qian, X., Dougherty, E., and Arróyave, R. (2018). Autonomous efficient experiment design for materials discovery with Bayesian model averaging. Physical Review Materials, 2, 113803.](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.2.113803)

[34. Niu, X., Chen, Y., Sun, M., Nagao, S., Aoki, Y., Niu, Z., and Zhang, L. (2025). Bayesian learning-assisted catalyst discovery for efficient iridium utilization in electrochemical water splitting. Science Advances, 11, eadw0894.](https://www.science.org/doi/10.1126/sciadv.adw0894)

[35. Wang, G., Mine, S., Chen, D., Jing, Y., Ting, K.W., Yamaguchi, T., Takao, M., Maeno, Z., Takigawa, I., Matsushita, K., et al. (2023). Accelerated discovery of multi-elemental reverse water-gas shift catalysts using extrapolative machine learning approach. Nature Communications, 14, 5861.](https://www.nature.com/articles/s41467-023-41341-3)

[36. Farache, D.E., Verduzco, J.C., McClure, Z.D., Desai, S., and Strachan, A. (2022). Active learning and molecular dynamics simulations to find high melting temperature alloys. Computational Materials Science, 209, 111386.](https://www.nature.com/articles/s41467-023-41341-3)

[37. Harwani, M., Verduzco, J.C., Lee, B.H., and Strachan, A. (2025). Accelerating active learning materials discovery with fair data and workflows: A case study for alloy melting temperatures. Computational Materials Science, 249, 113640.](https://www.sciencedirect.com/science/article/abs/pii/S0927025624008619)

[38. Nie, S., Xiang, Y., Wu, L., Lin, G., Liu, Q., Chu, S., and Wang, X. (2024). Active learning guided discovery of high entropy oxides featuring high H2-production. Journal of the American Chemical Society, 146, 29325–29334.](https://pubs.acs.org/doi/10.1021/jacs.4c06272)

[39. Cao, B., Su, T., Yu, S., Li, T., Zhang, T., Zhang, J., Dong, Z., and Zhang, T.Y. (2024). Active learning accelerates the discovery of high strength and high ductility lead-free solder alloys. Materials & Design, 241, 112921.](https://www.sciencedirect.com/science/article/pii/S0264127524002946)

[40. Jablonka, K.M., Jothiappan, G.M., Wang, S., Smit, B., and Yoo, B. (2021). Bias free multiobjective active learning for materials design and discovery. Nature Communications, 12, 2312.](https://www.nature.com/articles/s41467-021-22437-0)

[41. Xu, S., Chen, Z., Qin, M., Cai, B., Li, W., Zhu, R., Xu, C., and Xiang, X.D. (2024). Developing new electrocatalysts for oxygen evolution reaction via high throughput experiments and artificial intelligence. npj Computational Materials, 10, 194.](https://www.nature.com/articles/s41524-024-01386-4)

[42. Behler, J., and Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. Physical Review Letters, 98, 146401.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401)

[43. Smith, J.S., Isayev, O., and Roitberg, A.E. (2017). ANI-1: An extensible neural network potential with DFT accuracy at force field computational cost. Chemical Science, 8, 3192–3203.](https://pubs.rsc.org/en/content/articlelanding/2017/sc/c6sc05720a)

[44. Liu, Y., Kelley, K.P., Vasudevan, R.K., Funakubo, H., Ziatdinov, M.A., and Kalinin, S.V.(2022). Experimental discovery of structure–property relationships in ferroelectric materials via active learning. Nature Machine Intelligence 4, 341–350. ](https://www.nature.com/articles/s42256-022-00460-0)

[45. Bulanadi, R., Chowdhury, J., Hiroshi, F., Ziatdinov, M., Vasudevan, R., Biswas, A., and Liu,Y. (2025). Beyond optimization: Exploring novelty discovery in autonomous experiments.arXiv preprint arXiv:2508.20254.](https://www.researchgate.net/publication/395034231_Beyond_Optimization_Exploring_Novelty_Discovery_in_Autonomous_Experiments)

[46. Reiser, P., Neubert, M., Eberhard, A., Torresi, L., Zhou, C., Shao, C., Metni, H., van Hoesel, C., Schopmans, H., Sommer, T., et al. (2022). Graph neural networks for materials science and chemistry. Communications Materials, 3, 93.](https://www.nature.com/articles/s43246-022-00315-6)

[47. Chen, C., Ye, W., Zuo, Y., Zheng, C., and Ong, S.P. (2019). Graph networks as a universal machine learning framework for molecules and crystals. Chemistry of Materials, 31, 3564–3572.](https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294)

[48. Gal, Y., and Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning, PMLR, 1050–1059.](https://proceedings.mlr.press/v48/gal16.html)

[49. Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems, 30.](https://dl.acm.org/doi/10.5555/3295222.3295387)

[50. Xie, T., and Grossman, J.C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical Review Letters, 120, 145301.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)

[51. Ge, X., Yin, J., Ren, Z., Yan, K., Jing, Y., Cao, Y., Fei, N., Liu, X., Wang, X., Zhou, X., et al. (2024). Atomic design of alkyne semihydrogenation catalysts via active learning. Journal of the American Chemical Society, 146, 4993–5004.](https://pubs.acs.org/doi/10.1021/jacs.3c14495)

[52. Chun, H., Lunger, J.R., Kang, J.K., Gómez-Bombarelli, R., and Han, B. (2024). Active learning accelerated exploration of single-atom local environments in multimetallic systems for oxygen electrocatalysis. npj Computational Materials, 10, 246.](https://www.nature.com/articles/s41524-024-01432-1)

[53. Xu, W., Diesen, E., He, T., Reuter, K., and Margraf, J.T. (2024). Discovering high entropy alloy electrocatalysts in vast composition spaces with multiobjective optimization. Journal of the American Chemical Society, 146, 7698–7707.](https://pubs.acs.org/doi/10.1021/jacs.3c14486)

[54. Vasudevan, R.K., Kelley, K.P., Hinkle, J., Funakubo, H., Jesse, S., Kalinin, S.V., and Ziatdinov, M. (2021). Autonomous experiments in scanning probe microscopy and spectroscopy:choosing where to explore polarization dynamics in ferroelectrics. ACS nano 15, 11253– 11262.](https://pubs.acs.org/doi/10.1021/acsnano.0c10239)

[56. Ziatdinov, M., Liu, Y., Kelley, K., Vasudevan, R., and Kalinin, S.V. (2022). Bayesian active learning for scanning probe microscopy: From gaussian processes to hypothesis learning.ACS nano 16, 13492–13512.](https://pubs.acs.org/doi/10.1021/acsnano.2c05303)

[56. Lubbers, N., Lookman, T., and Barros, K. (2017). Inferring low-dimensional microstructure representations using convolutional neural networks. Physical Review E, 96, 052111.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.052111)

[57. DeCost, B.L., Lei, B., Francis, T., and Holm, E.A. (2019). High throughput quantitative metallography for complex microstructures using deep learning: A case study in ultrahigh carbon steel. Microscopy and Microanalysis, 25, 21–29.](https://academic.oup.com/mam/article-abstract/25/1/21/6887488?redirectedFrom=fulltext)

[58. Mozaffari, M.H., and Tay, L.L. (2021). Raman spectral analysis of mixtures with one-dimensional convolutional neural network. arXiv preprint arXiv:2106.05316.](https://ieeexplore.ieee.org/document/9664686)

[59. Szymanski, N.J., Bartel, C.J., Zeng, Y., Diallo, M., Kim, H., and Ceder, G. (2023). Adaptively driven x-ray diffraction guided by machine learning for autonomous phase identification. npj Computational Materials 9, 31.](https://www.nature.com/articles/s41524-023-00984-y)

[60. Tang, W.T., Chakrabarty, A., and Paulson, J.A. (2024). Beacon: A bayesian optimization strategy for novelty search in expensive black-box systems. arXiv:2406.03616.](https://www.semanticscholar.org/paper/BEACON%3A-A-Bayesian-Optimization-Strategy-for-Search-Tang-Chakrabarty/b15fa30730cec667142970019d34e7ab91bd08b7)

[61. Kingma, D.P., and Welling, M. (2014). Auto-encoding variational Bayes. stat, 1050, 1.](https://openreview.net/forum?id=33X9fd2-9FyZd)

[62. Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27.](https://proceedings.neurips.cc/paper/2022)

[63. Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6, 182–197.](https://ieeexplore.ieee.org/document/996017)

[64. Raju, R.K., Sivakumar, S., Wang, X., and Ulissi, Z.W. (2023). Cluster-MLP: An active learning genetic algorithm framework for accelerated discovery of global minimum configurations of pure and alloyed nanoclusters. Journal of Chemical Information and Modeling, 63, 6192–6197.](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01431)

[65. Wu, S., Hamel, C.M., Ze, Q., Yang, F., Qi, H.J., and Zhao, R. (2020). Evolutionary algorithm-guided voxel-encoding printing of functional hard-magnetic soft active materials. Advanced Intelligent Systems, 2, 2000060.](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202000060)

[66. Wolters, M.A. (2015). A genetic algorithm for selection of fixed-size subsets with application to design problems. Journal of Statistical Software 68, 1–18.](https://www.jstatsoft.org/article/view/v068c01)

[67. Liu, Q., Allamanis, M., Brockschmidt, M., and Gaunt, A. (2018). Constrained graph variational autoencoders for molecule design. Advances in Neural Information Processing Systems, 31.](https://arxiv.org/abs/1805.09076)

[68. Xie, T., Fu, X., Ganea, O., Barzilay, R., and Jaakkola, T. (2022). Crystal diffusion variational autoencoder for periodic material generation. Bulletin of the American Physical Society, 67.](https://openreview.net/forum?id=03RLpj-tc_)

[69. Lim, J., Ryu, S., Kim, J.W., and Kim, W.Y. (2018). Molecular generative model based on conditional variational autoencoder for de novo molecular design. Journal of Cheminformatics, 10, 31.](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0286-7)

[70. Xin, R., Siriwardane, E.M., Song, Y., Zhao, Y., Louis, S.Y., Nasiri, A., and Hu, J. (2021). Active-learning-based generative design for the discovery of wide-band-gap materials. The Journal of Physical Chemistry C, 125, 16118–16128.](https://pubs.acs.org/doi/10.1021/acs.jpcc.1c02438)

[71. Guo, W., Li, F., Wang, L., Zhu, L., Ye, Y., Wang, Z., Yang, B., Zhang, S., and Bai, S. (2025). Accelerated discovery of near-zero ablation ultra-high temperature ceramics via GAN-enhanced directionally constrained active learning. Advanced Powder Materials, 4, 100287.](https://www.sciencedirect.com/science/article/pii/S2772834X25000235)

[72. Kim, M., Ha, M.Y., Jung, W.B., Yoon, J., Shin, E., Kim, I.d., Lee, W.B., Kim, Y., and Jung, H.t. (2022). Searching for an optimal multi-metallic alloy catalyst by active learning combined with experiments. Advanced Materials, 34, 2108900.](https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/adma.202108900)

[73. Mok, D.H., Li, H., Zhang, G., Lee, C., Jiang, K., and Back, S. (2023). Data-driven discovery of electrocatalysts for CO2 reduction using active motifs-based machine learning. Nature Communications, 14, 7303.](https://www.nature.com/articles/s41467-023-43118-0)

[74. Gómez-Bombarelli, R., Aguilera-Iparraguirre, J., Hirzel, T.D., Duvenaud, D., Maclaurin, D., Blood-Forsythe, M.A., Chae, H.S., Einzinger, M., Ha, D.G., Wu, T., et al. (2016). Design of efficient molecular organic light-emitting diodes by a high-throughput virtual screening and experimental approach. Nature Materials, 15, 1120–1127.](https://www.nature.com/articles/nmat4717)

[75. Zhu, J.Y., Park, T., Isola, P., and Efros, A.A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision. pp. 2223–2232.](https://ieeexplore.ieee.org/document/8237506)

[76. Panisilvam, J., Hajizadeh, E., Weeratunge, H., Bailey, J., and Kim, S. (2023). Asymmetric cyclegans for inverse design of photonic metastructures. APL Machine Learning 1.](https://pubs.aip.org/aip/aml/article/1/4/046105/2918963/Asymmetric-CycleGANs-for-inverse-design-of)

[77. Bran, A.M., Cox, S., Schilter, O., Baldassari, C., White, A.D., and Schwaller, P. (2024). Augmenting large language models with chemistry tools. Nature Machine Intelligence, 6, 525–535.](https://www.nature.com/articles/s42256-024-00832-8)

[78. Xie, T., Wan, Y., Liu, Y., Zeng, Y., Wang, S., Zhang, W., Grazian, C., Kit, C., Ouyang, W., Zhou, D., et al. (2025). Large language models as materials science adapted learners. Nature.](https://openreview.net/forum?id=iTjHGQweoF)

[79. Boiko, D.A., MacKnight, R., Kline, B., and Gomes, G. (2023). Autonomous chemical research with large language models. Nature, 624, 570–578.](https://www.nature.com/articles/s41586-023-06792-0)

[80. Buehler, M.J. (2025). Preflexor: Preference-based recursive language modeling for exploratory optimization of reasoning and agentic thinking. npj Artificial Intelligence 1, 4.](https://www.nature.com/articles/s44387-025-00003-z)

[81. Buehler, M.J. (2024). Cephalo: Multi-modal vision-language models for bio-inspired materials analysis and design. Advanced Functional Materials 34, 2409531.](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adfm.202409531)

[82. Buehler, M.J. (2025). In situ graph reasoning and knowledge expansion using graph-preflexor. Advanced Intelligent Discovery pp. 202500006.](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aidi.202500006)

[83. Zhang, Y., Han, Y., Chen, S., Yu, R., Zhao, X., Liu, X., Zeng, K., Yu, M., Tian, J., Zhu, F., et al. (2025). Large language models to accelerate organic chemistry synthesis. Nature Machine Intelligence, 1–13.](https://www.nature.com/articles/s42256-025-01066-y)

[84. Ghafarollahi, A., and Buehler, M.J. (2024). Protagents: protein discovery via large language model multi-agent collaborations combining physics and machine learning. Digital950
Discovery 3, 1389–1409.](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00013g)

[85. Ghafarollahi, A., and Buehler, M.J. (2025). Sparks: Multi-agent artificial intelligence model discovers protein design principles. arXiv preprint arXiv:2504.19017.](https://arxiv.org/abs/2504.19017)

[86. Snoek, J., Larochelle, H., and Adams, R.P. (2012). Practical Bayesian optimization of machine learning algorithms. Advances in Neural Information Processing Systems, 25.](https://dl.acm.org/doi/10.5555/2999325.2999464)

[87. Asprion, N., Böttcher, R., Pack, R., Stavrou, M.E., Höller, J., Schwientek, J., and Bortz, M. (2019). Gray-box modeling for the optimization of chemical processes. Chemie Ingenieur Technik, 91, 305–313.](https://onlinelibrary.wiley.com/doi/abs/10.1002/cite.201800086)

[88. Psichogios, D.C., and Ungar, L.H. (1992). A hybrid neural network-first principles approach to process modeling. AIChE Journal, 38, 1499–1511.](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690381003)

[89. Molga, E. (2003). Neural network approach to support modelling of chemical reactors: Problems, resolutions, criteria of application. Chemical Engineering and Processing: Process Intensification, 42, 675–695.](https://www.sciencedirect.com/science/article/abs/pii/S0255270102002052)

[90. Ziatdinov, M.A., Ghosh, A., and Kalinin, S.V. (2022). Physics makes the difference: Bayesian optimization and active learning via augmented Gaussian process. Machine Learning: Science and Technology, 3, 015003.](https://iopscience.iop.org/article/10.1088/2632-2153/ac4baa)

[91. Ladygin, V., Beniya, I., Makarov, E., and Shapeev, A. (2021). Bayesian learning of thermodynamic integration and numerical convergence for accurate phase diagrams. Physical Review B, 104, 104102.](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.104102)

[92. Vela, B., Khatamsaz, D., Acemi, C., Karaman, I., and Arróyave, R. (2023). Data-augmented modeling for yield strength of refractory high entropy alloys: A Bayesian approach. Acta Materialia, 261, 119351.](https://www.sciencedirect.com/science/article/abs/pii/S135964542300681X)

[93. Khatamsaz, D., Neuberger, R., Roy, A.M., Zadeh, S.H., Otis, R., and Arróyave, R. (2023). A physics-informed Bayesian optimization approach for material design: Application to NiTi shape memory alloys. npj Computational Materials, 9, 221.](https://www.nature.com/articles/s41524-023-01173-7)

[94. Zhao, C., Zhang, F., Lou, W., Wang, X., and Yang, J. (2024). A comprehensive review of advances in physics-informed neural networks and their applications in complex fluid977
dynamics. Physics of Fluids 36.](https://pubs.aip.org/aip/pof/article-abstract/36/10/101301/3315125/A-comprehensive-review-of-advances-in-physics?redirectedFrom=fulltext)

[95. Anagnostopoulos, S.J., Toscano, J.D., Stergiopulos, N., and Karniadakis, G.E. (2025). Learning in pinns: Phase transition, diffusion equilibrium, and generalization. Neural Net-works pp. 107983.](https://www.sciencedirect.com/science/article/pii/S0893608025008640)

[96. Gebauer, N., Gastegger, M., and Sch¨ utt, K. (2019). Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules. Advances in neural information processing systems 32.](https://proceedings.neurips.cc/paper_files/paper/2019/file/a4d8e2a7e0d0c102339f97716d2fdfb6-Paper.pdf)

[97. Xie, T., Fu, X., Ganea, O., Barzilay, R., and Jaakkola, T. (2022). Crystal diffusion variational autoencoder for periodic material generation. Bulletin of the American Physical Society 67.](https://openreview.net/pdf?id=03RLpj-tc_) 

[98. Buehler, M.J. (2022). Modeling atomistic dynamic fracture mechanisms using a progressive transformer diffusion model. Journal of Applied Mechanics 89, 121009.](https://dspace.mit.edu/handle/1721.1/148574) 

[99. Gelbart, M.A., Snoek, J., and Adams, R.P. (2014). Bayesian optimization with unknown constraints. In 30th Conference on Uncertainty in Artificial Intelligence, UAI 2014, AUAI Press, 250–259.](https://dl.acm.org/doi/10.5555/3020751.3020778)

[100. Harris, S.B., Vasudevan, R., and Liu, Y. (2025). Active oversight and quality control in standard bayesian optimization for autonomous experiments. npj Computational Materials 11, 23.](https://www.nature.com/articles/s41524-024-01485-2)

[101. Sun, S., Tiihonen, A., Oviedo, F., Liu, Z., Thapa, J., Zhao, Y., Hartono, N.T.P., Goyal, A., Heumueller, T., Batali, C., et al. (2021). A data fusion approach to optimize compositional stability of halide perovskites. Matter, 4, 1305–1322.](https://www.sciencedirect.com/science/article/pii/S2590238521000084)

[102. Liu, Z., Rolston, N., Flick, A.C., Colburn, T.W., Ren, Z., Dauskardt, R.H., and Buonassisi, T. (2022). Machine learning with knowledge constraints for process optimization of open-air perovskite solar cell manufacturing. Joule, 6, 834–849.](https://www.sciencedirect.com/science/article/pii/S2542435122001301)

[103. Kusne, A.G., Yu, H., Wu, C., Zhang, H., Hattrick-Simpers, J., DeCost, B., Sarker, S., Oses, C., Toher, C., Curtarolo, S., et al. (2020). On-the-fly closed-loop materials discovery via Bayesian active learning. Nature Communications, 11, 5966.](https://www.nature.com/articles/s41467-020-19597-w)

[104. Tian, Y., Li, T., Pang, J., Zhou, Y., Xue, D., Ding, X., and Lookman, T. (2025). Materials design with target-oriented Bayesian optimization. npj Computational Materials, 11, 209.](https://www.nature.com/articles/s41524-025-01704-4)

[105. Vlachos, A. (2008). A stopping criterion for active learning. Computer Speech & Language, 22, 295–312.](https://www.sciencedirect.com/science/article/abs/pii/S088523080700068X)

[106. Bloodgood, M., and Vijay-Shanker, K. (2009). A method for stopping active learning based on stabilizing predictions and the need for user-adjustable stopping. In Thirteenth Conference on Computational Natural Language Learning (CoNLL), 39.](https://aclanthology.org/W09-1107/)

[107. Ishibashi, H., and Hino, H. (2020). Stopping criterion for active learning based on deterministic generalization bounds. In International Conference on Artificial Intelligence and Statistics, PMLR, 386–397.](https://proceedings.mlr.press/v108/ishibashi20a.html)

[108. Pullar-Strecker, Z., Dost, K., Frank, E., and Wicker, J. (2024). Hitting the target: Stopping active learning at the cost-based optimum. Machine Learning, 113, 1529–1547.](https://link.springer.com/article/10.1007/s10994-022-06253-1)

[109. Callaghan, M.W., and Müller-Hansen, F. (2020). Statistical stopping criteria for automated screening in systematic reviews. Systematic Reviews, 9, 273.](https://systematicreviewsjournal.biomedcentral.com/articles/10.1186/s13643-020-01521-4)

[110. Boetje, J., and van de Schoot, R. (2024). The SAFE procedure: A practical stopping heuristic for active learning-based screening in systematic reviews and meta-analyses. Systematic Reviews, 13, 81.](https://systematicreviewsjournal.biomedcentral.com/articles/10.1186/s13643-024-02502-7)

[111. Sholokhov, A., Liu, Y., Mansour, H., and Nabi, S. (2023). Physics-informed neural ode (pinode): embedding physics into models using collocation points. Scientific Reports 13, 10166.](https://www.nature.com/articles/s41598-023-36799-6)

[112. Qin, X., Zhong, B., Lv, S., Long, X., Xu, H., Li, L., Xu, K., Lou, Z., Luo, Q., and Wang, L.(2024). A zero-voltage-writing artificial nervous system based on biosensor integrated on ferroelectric tunnel junction. Advanced Materials 36, 2404026.](https://onlinelibrary.wiley.com/doi/abs/10.1002/adma.202404026)

[113. Li, Z., Xu, H., Zheng, Y., Liu, L., Li, L., Lou, Z., and Wang, L. (2025). A reconfigurable heterostructure transistor array for monocular 3d parallax reconstruction. Nature Electronics 8, 46–55.](https://www.nature.com/articles/s41928-024-01261-6)

[114. Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J.P., Kornbluth, M., Molinari, N.,Smidt, T.E., and Kozinsky, B. (2022). E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature communications 13, 2453.](https://www.nature.com/articles/s41467-022-29939-5)

[115. Buehler, M.J. (2022). Prediction of atomic stress fields using cycle-consistent adversarial neural networks based on unpaired and unmatched sparse datasets. Materials Advances 3, 6280–6290.](https://pubs.rsc.org/en/content/articlelanding/2022/ma/d2ma00223j)

[116. Kandasamy, K., Dasarathy, G., Oliva, J.B., Schneider, J., and P´ oczos, B. (2016). Gaussian process bandit optimisation with multi-fidelity evaluations. Advances in neural information processing systems 29.](https://proceedings.neurips.cc/paper_files/paper/2016/file/605ff764c617d3cd28dbbdd72be8f9a2-Paper.pdf)

[117. Leonov, A.I., Hammer, A.J., Lach, S., Mehr, S.H.M., Caramelli, D., Angelone, D., Khan, A., O’Sullivan, S., Craven, M., Wilbraham, L., et al. (2024). An integrated self-optimizing programmable chemical synthesis and reaction engine. Nature Communications, 15, 1240.](https://www.nature.com/articles/s41467-024-45444-3)

[118. Rauschen, R., Ayme, J.F., Matysiak, B.M., Thomas, D., and Cronin, L. (2025). A programmable modular robot for the synthesis of molecular machines. Chem.](https://www.sciencedirect.com/science/article/pii/S2451929425000944)

[119. Gao, F., Li, H., Chen, Z., Yi, Y., Nie, S., Cheng, Z., Liu, Z., Guo, Y., Liu, S., Qin, Q., et al. (2025). A chemical autonomous robotic platform for end-to-end synthesis of nanoparticles. Nature Communications, 16, 7558.](https://www.nature.com/articles/s41467-025-62994-2)

