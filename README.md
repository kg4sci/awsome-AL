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
* Design spaces are vast: e.g., 10^60 possible high-entropy alloys from 5+ elements.
* AL reduces queries by 50-90% (e.g., mapping performance landscapes with 10s vs. 100s of experiments).

## Application of AL in the field of materials
We introduces a task-oriented classification of Active Learning (AL) methods, emphasizing their diversity in reducing labeling costs and speeding up discovery. It organizes AL along two orthogonal axes: acquisition paradigms (pool-based vs. generative) and knowledge integration (purely data-driven vs. domain-informed). This framework provides a conceptual foundation, distinguishing what to optimize (objectives) from how candidates are sourced (acquisition), and unifies variants under an iterative acquisition-update process tailored to materials science constraints.
### Pool-Based Active Learning for Materials Science

### Generative Active Learning for Materials Science

## References
