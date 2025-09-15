# Accelerating Materials Discovery through Active Learning: Methods, Challenges and Opportunities
## Abstract
Recent advances in high-throughput experimentation and computational techniques have ushered materials science into a data-driven era, yet traditional empirical and blind screening approaches struggle to keep pace with the vast compositional complexity and multi-scale mechanisms of modern materials discovery. High costs and time-intensive processes, such as synthesis experiments and density functional theory (DFT) calculations, underscore the need for intelligent strategies to optimize resource use. Bayesian optimization and active learning (AL) have emerged as powerful tools to accelerate materials discovery by efficiently exploring parameter spaces and prioritizing the most informative samples for labeling. AL, in particular, reduces experimental and computational costs by selectively targeting samples that maximize model improvement, enabling rapid mapping of performance landscapes with significantly fewer experiments. This review introduces a novel classification of AL methods based on interaction and sampling scenarios, aligning with real-world experimental and computational conditions. It critically examines challenges, such as integrating domain knowledge and ensuring interpretable decision-making, while providing a practical roadmap for implementing AL in resource-constrained materials research environments.
## Key Themes
* **Core Problem**:Materials discovery faces exponential combinatorial complexity (e.g., multi-component alloys) and multi-scale physics, making trial-and-error inefficient.
* **AL Solution**:AL prioritizes "informative" samples for labeling, forming closed-loop workflows with surrogate models, query strategies, and experimental/computational oracles.
* **Applications**: Compositional selection, structural design, processing optimization, and property prediction.
* **Innovations**:Unified mathematical formulation for pool- and generative-based AL; emphasis on knowledge-integrated methods.
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
We introduces a task-oriented classification of Active Learning (AL) methods, emphasizing their diversity in reducing labeling costs and speeding up discovery. It organizes AL along two orthogonal axes: acquisition paradigms (pool-based vs. generative) and knowledge integration (purely data-driven vs. domain-informed). This framework provides a conceptual foundation, distinguishing what to optimize (objectives) from how candidates are sourced (acquisition), and unifies variants under an iterative acquisition-update process tailored to materials science constraints.
### Pool-Based Active Learning for Materials Science

| Document subsection (original) | Primary material task types | Brief rationale |
| :-----| :----- | :----- |
| Probabilistic and Bayesian Models | Composition screening; property prediction; multi-objective optimization | GP and RF provide predictive mean and variance directly, which enables acquisition functions (e.g., EI, UCB) and constrained/multi-objective extensions. |
| Neural Networks | High-dimensional property prediction; surrogate models for potential-energy surfaces; image/spectral characterization | DNNs (MLP/CNN/HDNNP) can fit complex nonlinear mappings, handle structured inputs (images, spectra), and act as low-latency surrogates replacing expensive simulations.|
| Graph Neural Networks (GNNs) | Crystal/molecular structure design and discovery; structure–composition joint screening; catalyst/functional-material prediction | GNNs encode atom–bond topology naturally, making them well-suited for structure-centric tasks and model–DFT closed-loop screening.


### Generative Active Learning for Materials Science

| Document subsection (original) | Primary material task types | Brief rationale |
| :-----| :----- | :----- |
| Evolutionary Algorithm-Based Generative AL | Combinatorial/discrete design exploration; multi-objective (Pareto) optimization | Evolutionary operators (crossover, mutation) directly manipulate discrete design variables and support Pareto-front construction via non-dominated sorting. |
| VAE-Based Generative AL | Conditional/target-oriented generation; latent-space optimization; virtual library expansion | VAEs provide continuous, differentiable latent representations that facilitate interpolation, optimization, and conditional sampling for target-directed design. |
| GAN-Based Generative AL | High-diversity candidate generation; exploratory discovery; few-shot target search | GANs excel at producing diverse samples for exploratory search but typically require physics-based constraints or downstream filtering to ensure physical feasibility. |
| LLMs-Based AL | Textual knowledge extraction; generative proposal of compositions/structures/recipes; human-machine collaboration | LLMs serve as versatile generators that formalize priors from literature and suggest candidates/strategies, accelerating discovery in label-poor settings.|

## References

[Batra, R., Song, L., and Ramprasad, R. (2021). Emerging materials intelligence ecosystems propelled by machine learning. Nature Reviews Materials, 6, 655–678.]()

[Gubernatis, J., and Lookman, T. (2018). Machine learning in materials design and discovery: Examples from the present and suggestions for the future. Physical Review Materials, 2, 120301.]()

[Liu, Y., Zhao, T., Ju, W., and Shi, S. (2017). Materials discovery and design using machine learning. Journal of Materiomics, 3, 159–177.]()

[Slattery, A., Wen, Z., Tenblad, P., Sanjosé-Orduna, J., Pintossi, D., den Hartog, T., and Noël, T. (2024). Automated self-optimization, intensification, and scale-up of photocatalysis in flow. Science, 383, eadj1817.]()

[Lookman, T., Balachandran, P.V., Xue, D., and Yuan, R. (2019). Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design. npj Computational Materials, 5, 21.]()

[Bai, X., and Zhang, X. (2025). Artificial intelligence-powered materials science. Nano-Micro Letters, 17, 1–30.]()

[Shahriari, B., Swersky, K., Wang, Z., Adams, R.P., and De Freitas, N. (2015). Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE, 104, 148–175.]()

[Szymanski, N.J., Rendy, B., Fei, Y., Kumar, R.E., He, T., Milsted, D., McDermott, M.J., Gallant, M., Cubuk, E.D., Merchant, A., et al. (2023). An autonomous laboratory for the accelerated synthesis of novel materials. Nature, 624, 86–91.]()

[MacLeod, B.P., Parlane, F.G., Morrissey, T.D., Häse, F., Roch, L.M., Dettelbach, K.E., Moreira, R., Yunker, L.P., Rooney, M.B., Deeth, J.R., et al. (2020). Self-driving laboratory for accelerated discovery of thin-film materials. Science Advances, 6, eaaz8867.]()

[Settles, B. (2009). Active learning literature survey. Science, 10, 237–304.]()

[Schmidt, J., Marques, M.R., Botti, S., and Marques, M.A. (2019). Recent advances and applications of machine learning in solid-state materials science. npj Computational Materials, 5, 83.]()

[Kumar, P., and Gupta, A. (2020). Active learning query strategies for classification, regression, and clustering: A survey. Journal of Computer Science and Technology, 35, 913–945.]()

[Saal, J.E., Oliynyk, A.O., and Meredig, B. (2020). Machine learning in materials discovery: Confirmed predictions and their underlying approaches. Annual Review of Materials Research, 50, 49–69.]()

[Yunfan, W., Yuan, T., Yumei, Z., and Dezhen, X. (2023). Progress on active learning assisted materials discovery. Journal of the Chinese Ceramic Society, 51, 544–551.]()

[Lewis, D.D. (1995). A sequential algorithm for training text classifiers: Corrigendum and additional data. ACM SIGIR Forum, 29, 13–19.]()

[Beluch, W.H., Genewein, T., Nürnberger, A., and Köhler, J.M. (2018). The power of ensembles for active learning in image classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 9368–9377.]()

[Zhu, M., Fan, C., Chen, H., Liu, Y., Mao, W., Xu, X., and Shen, C. (2024). Generative active learning for long-tailed instance segmentation. In International Conference on Machine Learning, PMLR, 62349–62368.]()

[Zhu, J.J., and Bento, J. (2017). Generative adversarial active learning. arXiv preprint arXiv:1702.07956.]()

[Liu, Y., Li, Z., Zhou, C., Jiang, Y., Sun, J., Wang, M., and He, X. (2019). Generative adversarial active learning for unsupervised outlier detection. IEEE Transactions on Knowledge and Data Engineering, 32, 1517–1528.]()

[Gui, J., Sun, Z., Wen, Y., Tao, D., and Ye, J. (2021). A review on generative adversarial networks: Algorithms, theory, and applications. IEEE Transactions on Knowledge and Data Engineering, 35, 3313–3332.]()

[Rao, Z., Tung, P.Y., Xie, R., Wei, Y., Zhang, H., Ferrari, A., Klaver, T., Körmann, F., Sukumar, P.T., Kwiatkowski da Silva, A., et al. (2022). Machine learning–enabled high-entropy alloy discovery. Science, 378, 78–85.]()

[Sohail, Y., Zhang, C., Xue, D., Zhang, J., Zhang, D., Gao, S., Yang, Y., Fan, X., Zhang, H., Liu, G., et al. (2025). Machine-learning design of ductile FeNiCoAlTa alloys with high strength. Nature, 1–6.]()

[Moon, J., Beker, W., Siek, M., Kim, J., Lee, H.S., Hyeon, T., and Grzybowski, B.A. (2024). Active learning guides discovery of a champion four-metal perovskite oxide for oxygen evolution electrocatalysis. Nature Materials, 23, 108–115.]()

[Suvarna, M., Zou, T., Chong, S.H., Ge, Y., Martín, A.J., and Pérez-Ramírez, J. (2024). Active learning streamlines development of high performance catalysts for higher alcohol synthesis. Nature Communications, 15, 5844.]()

[Merchant, A., Batzner, S., Schoenholz, S.S., Aykol, M., Cheon, G., and Cubuk, E.D. (2023). Scaling deep learning for materials discovery. Nature, 624, 80–85.]()

[Lee, J.A., Park, J., Sagong, M.J., Ahn, S.Y., Cho, J.W., Lee, S., and Kim, H.S. (2025). Active learning framework to optimize process parameters for additive-manufactured Ti-6Al-4V with high strength and ductility. Nature Communications, 16, 931.]()

[Zhang, R., Xu, J., Zhang, H., Xu, G., and Luo, T. (2025). Active learning-guided exploration of thermally conductive polymers under strain. Digital Discovery, 4, 812–823.]()

[Johnson, N.S., Mishra, A.A., Kirsch, D.J., and Mehta, A. (2024). Active learning for rapid targeted synthesis of compositionally complex alloys. Materials, 17, 4038.]()

[Verduzco, J.C., Marinero, E.E., and Strachan, A. (2021). An active learning approach for the design of doped LLZO ceramic garnets for battery applications. Integrating Materials and Manufacturing Innovation, 10, 299–310.]()

[Pedersen, J.K., Clausen, C.M., Krysiak, O.A., Xiao, B., Batchelor, T.A., Löffler, T., Mints, V.A., Banko, L., Arenz, M., Savan, A., et al. (2021). Bayesian optimization of high-entropy alloy compositions for electrocatalytic oxygen reduction. Angewandte Chemie, 133, 24346–24354.]()

[Talapatra, A., Boluki, S., Duong, T., Qian, X., Dougherty, E., and Arróyave, R. (2018). Autonomous efficient experiment design for materials discovery with Bayesian model averaging. Physical Review Materials, 2, 113803.]()

[Niu, X., Chen, Y., Sun, M., Nagao, S., Aoki, Y., Niu, Z., and Zhang, L. (2025). Bayesian learning-assisted catalyst discovery for efficient iridium utilization in electrochemical water splitting. Science Advances, 11, eadw0894.]()

[Wang, G., Mine, S., Chen, D., Jing, Y., Ting, K.W., Yamaguchi, T., Takao, M., Maeno, Z., Takigawa, I., Matsushita, K., et al. (2023). Accelerated discovery of multi-elemental reverse water-gas shift catalysts using extrapolative machine learning approach. Nature Communications, 14, 5861.]()

[Farache, D.E., Verduzco, J.C., McClure, Z.D., Desai, S., and Strachan, A. (2022). Active learning and molecular dynamics simulations to find high melting temperature alloys. Computational Materials Science, 209, 111386.]()

[Harwani, M., Verduzco, J.C., Lee, B.H., and Strachan, A. (2025). Accelerating active learning materials discovery with fair data and workflows: A case study for alloy melting temperatures. Computational Materials Science, 249, 113640.]()

[Nie, S., Xiang, Y., Wu, L., Lin, G., Liu, Q., Chu, S., and Wang, X. (2024). Active learning guided discovery of high entropy oxides featuring high H2-production. Journal of the American Chemical Society, 146, 29325–29334.]()

[Cao, B., Su, T., Yu, S., Li, T., Zhang, T., Zhang, J., Dong, Z., and Zhang, T.Y. (2024). Active learning accelerates the discovery of high strength and high ductility lead-free solder alloys. Materials & Design, 241, 112921.]()

[Jablonka, K.M., Jothiappan, G.M., Wang, S., Smit, B., and Yoo, B. (2021). Bias free multiobjective active learning for materials design and discovery. Nature Communications, 12, 2312.]()

[Xu, S., Chen, Z., Qin, M., Cai, B., Li, W., Zhu, R., Xu, C., and Xiang, X.D. (2024). Developing new electrocatalysts for oxygen evolution reaction via high throughput experiments and artificial intelligence. npj Computational Materials, 10, 194.]()

[Behler, J., and Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. Physical Review Letters, 98, 146401.]()

[Smith, J.S., Isayev, O., and Roitberg, A.E. (2017). ANI-1: An extensible neural network potential with DFT accuracy at force field computational cost. Chemical Science, 8, 3192–3203.]()

[Lubbers, N., Lookman, T., and Barros, K. (2017). Inferring low-dimensional microstructure representations using convolutional neural networks. Physical Review E, 96, 052111.]()

[DeCost, B.L., Lei, B., Francis, T., and Holm, E.A. (2019). High throughput quantitative metallography for complex microstructures using deep learning: A case study in ultrahigh carbon steel. Microscopy and Microanalysis, 25, 21–29.]()

[Reiser, P., Neubert, M., Eberhard, A., Torresi, L., Zhou, C., Shao, C., Metni, H., van Hoesel, C., Schopmans, H., Sommer, T., et al. (2022). Graph neural networks for materials science and chemistry. Communications Materials, 3, 93.]()

[Chen, C., Ye, W., Zuo, Y., Zheng, C., and Ong, S.P. (2019). Graph networks as a universal machine learning framework for molecules and crystals. Chemistry of Materials, 31, 3564–3572.]()

[Gal, Y., and Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning, PMLR, 1050–1059.]()

[Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems, 30.]()

[Xie, T., and Grossman, J.C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical Review Letters, 120, 145301.]()

[Ge, X., Yin, J., Ren, Z., Yan, K., Jing, Y., Cao, Y., Fei, N., Liu, X., Wang, X., Zhou, X., et al. (2024). Atomic design of alkyne semihydrogenation catalysts via active learning. Journal of the American Chemical Society, 146, 4993–5004.]()

[Chun, H., Lunger, J.R., Kang, J.K., Gómez-Bombarelli, R., and Han, B. (2024). Active learning accelerated exploration of single-atom local environments in multimetallic systems for oxygen electrocatalysis. npj Computational Materials, 10, 246.]()

[Xu, W., Diesen, E., He, T., Reuter, K., and Margraf, J.T. (2024). Discovering high entropy alloy electrocatalysts in vast composition spaces with multiobjective optimization. Journal of the American Chemical Society, 146, 7698–7707.]()

[Mozaffari, M.H., and Tay, L.L. (2021). Raman spectral analysis of mixtures with one-dimensional convolutional neural network. arXiv preprint arXiv:2106.05316.]()

[Kingma, D.P., and Welling, M. (2014). Auto-encoding variational Bayes. stat, 1050, 1.]()

[Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27.]()

[Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6, 182–197.]()

[Raju, R.K., Sivakumar, S., Wang, X., and Ulissi, Z.W. (2023). Cluster-MLP: An active learning genetic algorithm framework for accelerated discovery of global minimum configurations of pure and alloyed nanoclusters. Journal of Chemical Information and Modeling, 63, 6192–6197.]()

[Wu, S., Hamel, C.M., Ze, Q., Yang, F., Qi, H.J., and Zhao, R. (2020). Evolutionary algorithm-guided voxel-encoding printing of functional hard-magnetic soft active materials. Advanced Intelligent Systems, 2, 2000060.]()

[Liu, Q., Allamanis, M., Brockschmidt, M., and Gaunt, A. (2018). Constrained graph variational autoencoders for molecule design. Advances in Neural Information Processing Systems, 31.]()

[Xie, T., Fu, X., Ganea, O., Barzilay, R., and Jaakkola, T. (2022). Crystal diffusion variational autoencoder for periodic material generation. Bulletin of the American Physical Society, 67.]()

[Lim, J., Ryu, S., Kim, J.W., and Kim, W.Y. (2018). Molecular generative model based on conditional variational autoencoder for de novo molecular design. Journal of Cheminformatics, 10, 31.]()

[Xin, R., Siriwardane, E.M., Song, Y., Zhao, Y., Louis, S.Y., Nasiri, A., and Hu, J. (2021). Active-learning-based generative design for the discovery of wide-band-gap materials. The Journal of Physical Chemistry C, 125, 16118–16128.]()

[Guo, W., Li, F., Wang, L., Zhu, L., Ye, Y., Wang, Z., Yang, B., Zhang, S., and Bai, S. (2025). Accelerated discovery of near-zero ablation ultra-high temperature ceramics via GAN-enhanced directionally constrained active learning. Advanced Powder Materials, 4, 100287.]()

[Kim, M., Ha, M.Y., Jung, W.B., Yoon, J., Shin, E., Kim, I.d., Lee, W.B., Kim, Y., and Jung, H.t. (2022). Searching for an optimal multi-metallic alloy catalyst by active learning combined with experiments. Advanced Materials, 34, 2108900.]()

[Mok, D.H., Li, H., Zhang, G., Lee, C., Jiang, K., and Back, S. (2023). Data-driven discovery of electrocatalysts for CO2 reduction using active motifs-based machine learning. Nature Communications, 14, 7303.]()

[Gómez-Bombarelli, R., Aguilera-Iparraguirre, J., Hirzel, T.D., Duvenaud, D., Maclaurin, D., Blood-Forsythe, M.A., Chae, H.S., Einzinger, M., Ha, D.G., Wu, T., et al. (2016). Design of efficient molecular organic light-emitting diodes by a high-throughput virtual screening and experimental approach. Nature Materials, 15, 1120–1127.]()

[Bran, A.M., Cox, S., Schilter, O., Baldassari, C., White, A.D., and Schwaller, P. (2024). Augmenting large language models with chemistry tools. Nature Machine Intelligence, 6, 525–535.]()

[Xie, T., Wan, Y., Liu, Y., Zeng, Y., Wang, S., Zhang, W., Grazian, C., Kit, C., Ouyang, W., Zhou, D., et al. (2025). Large language models as materials science adapted learners. Nature.]()

[Boiko, D.A., MacKnight, R., Kline, B., and Gomes, G. (2023). Autonomous chemical research with large language models. Nature, 624, 570–578.]()

[Zhang, Y., Han, Y., Chen, S., Yu, R., Zhao, X., Liu, X., Zeng, K., Yu, M., Tian, J., Zhu, F., et al. (2025). Large language models to accelerate organic chemistry synthesis. Nature Machine Intelligence, 1–13.]()

[Snoek, J., Larochelle, H., and Adams, R.P. (2012). Practical Bayesian optimization of machine learning algorithms. Advances in Neural Information Processing Systems, 25.]()

[Asprion, N., Böttcher, R., Pack, R., Stavrou, M.E., Höller, J., Schwientek, J., and Bortz, M. (2019). Gray-box modeling for the optimization of chemical processes. Chemie Ingenieur Technik, 91, 305–313.]()

[Psichogios, D.C., and Ungar, L.H. (1992). A hybrid neural network-first principles approach to process modeling. AIChE Journal, 38, 1499–1511.]()

[Molga, E. (2003). Neural network approach to support modelling of chemical reactors: Problems, resolutions, criteria of application. Chemical Engineering and Processing: Process Intensification, 42, 675–695.]()

[Ziatdinov, M.A., Ghosh, A., and Kalinin, S.V. (2022). Physics makes the difference: Bayesian optimization and active learning via augmented Gaussian process. Machine Learning: Science and Technology, 3, 015003.]()

[Ladygin, V., Beniya, I., Makarov, E., and Shapeev, A. (2021). Bayesian learning of thermodynamic integration and numerical convergence for accurate phase diagrams. Physical Review B, 104, 104102.]()

[Vela, B., Khatamsaz, D., Acemi, C., Karaman, I., and Arróyave, R. (2023). Data-augmented modeling for yield strength of refractory high entropy alloys: A Bayesian approach. Acta Materialia, 261, 119351.]()

[Khatamsaz, D., Neuberger, R., Roy, A.M., Zadeh, S.H., Otis, R., and Arróyave, R. (2023). A physics-informed Bayesian optimization approach for material design: Application to NiTi shape memory alloys. npj Computational Materials, 9, 221.]()

[Gelbart, M.A., Snoek, J., and Adams, R.P. (2014). Bayesian optimization with unknown constraints. In 30th Conference on Uncertainty in Artificial Intelligence, UAI 2014, AUAI Press, 250–259.]()

[Sun, S., Tiihonen, A., Oviedo, F., Liu, Z., Thapa, J., Zhao, Y., Hartono, N.T.P., Goyal, A., Heumueller, T., Batali, C., et al. (2021). A data fusion approach to optimize compositional stability of halide perovskites. Matter, 4, 1305–1322.]()

[Liu, Z., Rolston, N., Flick, A.C., Colburn, T.W., Ren, Z., Dauskardt, R.H., and Buonassisi, T. (2022). Machine learning with knowledge constraints for process optimization of open-air perovskite solar cell manufacturing. Joule, 6, 834–849.]()

[Kusne, A.G., Yu, H., Wu, C., Zhang, H., Hattrick-Simpers, J., DeCost, B., Sarker, S., Oses, C., Toher, C., Curtarolo, S., et al. (2020). On-the-fly closed-loop materials discovery via Bayesian active learning. Nature Communications, 11, 5966.]()

[Tian, Y., Li, T., Pang, J., Zhou, Y., Xue, D., Ding, X., and Lookman, T. (2025). Materials design with target-oriented Bayesian optimization. npj Computational Materials, 11, 209.]()

[Vlachos, A. (2008). A stopping criterion for active learning. Computer Speech & Language, 22, 295–312.]()

[Bloodgood, M., and Vijay-Shanker, K. (2009). A method for stopping active learning based on stabilizing predictions and the need for user-adjustable stopping. In Thirteenth Conference on Computational Natural Language Learning (CoNLL), 39.]()

[Ishibashi, H., and Hino, H. (2020). Stopping criterion for active learning based on deterministic generalization bounds. In International Conference on Artificial Intelligence and Statistics, PMLR, 386–397.]()

[Pullar-Strecker, Z., Dost, K., Frank, E., and Wicker, J. (2024). Hitting the target: Stopping active learning at the cost-based optimum. Machine Learning, 113, 1529–1547.]()

[Callaghan, M.W., and Müller-Hansen, F. (2020). Statistical stopping criteria for automated screening in systematic reviews. Systematic Reviews, 9, 273.]()

[Boetje, J., and van de Schoot, R. (2024). The SAFE procedure: A practical stopping heuristic for active learning-based screening in systematic reviews and meta-analyses. Systematic Reviews, 13, 81.]()

[Leonov, A.I., Hammer, A.J., Lach, S., Mehr, S.H.M., Caramelli, D., Angelone, D., Khan, A., O’Sullivan, S., Craven, M., Wilbraham, L., et al. (2024). An integrated self-optimizing programmable chemical synthesis and reaction engine. Nature Communications, 15, 1240.]()

[Rauschen, R., Ayme, J.F., Matysiak, B.M., Thomas, D., and Cronin, L. (2025). A programmable modular robot for the synthesis of molecular machines. Chem.]()

[Gao, F., Li, H., Chen, Z., Yi, Y., Nie, S., Cheng, Z., Liu, Z., Guo, Y., Liu, S., Qin, Q., et al. (2025). A chemical autonomous robotic platform for end-to-end synthesis of nanoparticles. Nature Communications, 16, 7558.]()

