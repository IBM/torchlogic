# torchlogic Overview

The algorithms implemented in torchlogic aims to learn a set of 
logical rules from data that will classify, or rank, depending on the loss function.
The core technology to this algorithm is 
[Weighted Lukasiewicz Logic](https://arxiv.org/abs/2006.13155), which enables a Data Scientist
to define logic that can be used for a prediction task, and leverage data to learn weights
for that logic that minimize loss on that prediction task. 

Leveraging this technology, we develop learning
algorithms that mines useful rules from raw data.  The algorithm
implemented thus far is Bandit-Reinforced Neural Reasoning Network
(R-NRN).

NRNs are a fully explainable AI approach that enables the seamless integration between 
expert knowledge (e.g. sellers, sales leaders etc.) and patterns learned 
from data so that those who use its predictions can understand the predictive 
logic, learn from patterns detected in data, and influence future predictions by 
adding their own knowledge. Some key properties of this algorithm that make it especially suited to
enterprise application are as follows:

- Transparency: The logic learned by the algorithm is fully transparent.  It can be inspected, and for complex logic one can control the level of detail to surface during inspection -- limiting the exposed logic by order of importance.
- Domain Knowledge: The algorithm can be easily extended to make use of expert knowledge.  This knowledge can be encoded in the RN and used for predictions.  If the knowledge is not useful, it will be forgotten.
- Control: Enterprise application of AI often involve use of business rules.  Similar to included expert knowledge, one can introduce business rules to the RN in the form of encoded logic, and by fixing the level of importance for those rules, ensure that the model will obey them.
- Scalability: The algorithm performs well at all scales of data.  If you have 100 samples or 1M samples, 10 features or 10K features, the algorithm can effectively learn to classify.
- Performance: In many experiments, the algorithms performance can exceed that of traditional ML methods such as Random Forest by 5%-10%.