# Use Cases

As previously mentioned, Neural Reasoning Networks are especially suited
to achieve the following benefits:

- Transparency: The logic learned by the algorithm is fully transparent.  It can be inspected, and for complex logic one can control the level of detail to surface during inspection -- limiting the exposed logic by order of importance.
- Domain Knowledge: The algorithm can be easily extended to make use of expert knowledge.  This knowledge can be encoded in the RN and used for predictions.  If the knowledge is not useful, it will be forgotten.
- Control: Enterprise application of AI often involve use of business rules.  Similar to included expert knowledge, one can introduce business rules to the RN in the form of encoded logic, and by fixing the level of importance for those rules, ensure that the model will obey them.
- Scalability: The algorithm performs well at all scales of data.  If you have 100 samples or 1M samples, 10 features or 10K features, the algorithm can effectively learn to classify.
- Performance: In many experiments, the algorithms performance can exceed that of traditional ML methods such as Random Forest by 5%-10%.

These benefits lend themselves to certain use cases within the AI workflow.
Below are examples of when and how Reasoning Networks might be used.  
This list is certainly not exhaustive.

## Use Case 1:  Data Understanding

NRNs should likely not serve as an initial baseline model due to their
relatively higher difficulty in training compared to easily trained models such
as Random Forest.  However, they may be quite useful during model development
as a tool to understand ones data and debug a system.

NRNs unique model explanation capabilities can enable a Data Scientist
to train a model and understand directly how the data is used to generate
predictions.  For example, one can gain some basic understanding of
how to classify tumors from data by inspecting a trained NRN.

!!! Example "Benign Class"
     - Predicted Value: 0.7

     - The patient is in the benign class because:
         - ANY of the following are TRUE: 
            - ALL of the following are TRUE: 
                - the concave points error is greater than 0.037, 
                - AND the worst perimeter is NOT greater than 61.855, 
            - OR ALL of the following are TRUE: 
                - the mean radius is NOT greater than 12.073, 
                - AND the mean concavity is NOT greater than 0.195

!!! Example "Malignant Class"
    - Predicted Value: 0.3
     
    - The patient is in the negative of benign class because:
        - ANY of the following are TRUE: 
             - ALL of the following are TRUE: 
                 - the concave points error is NOT greater than 0.033, 
                 - AND the worst perimeter is greater than 64.666, 
                 - AND the worst symmetry is NOT greater than 0.374, 
             - OR ALL of the following are TRUE: 
                 - the mean radius is greater than 13.214, 
                 - AND the mean concavity is greater than 0.239

Without any prior knowledge of the data, one can begin to understand that
tumors with larger perimeters, a larger radius, lower symmetry and more 
concavity are likely to be tumors.  Furthermore, we can see some of the
critical boundaries in the data for predictions that are fairly confident
of either classification.

## Use Case 2: End User Transparency

In many industries, such as Business, Healthcare, Finance, or Law, the
ability for end users to understand a model's behavior is at the very least
a motivator for adopting model predictions, and in many cases a requirement
for using a model all together.

Reinforced Reasoning Networks enable Data Scientists to produce sample
level explanations of the model's predictions, which can be used to
directly understand the context of the prediction, increasing trust.

!!! Example "Malignant Class"
    - Sample 0: The patient was in the **negative** of benign class because:
        - ANY of the following are TRUE: 
             - ALL of the following are TRUE: 
                 - the concave points error is NOT greater than 0.05279, 
                 - AND the worst perimeter is greater than 56.23291, 
                 - AND the worst symmetry is NOT greater than 0.6638, 
             - OR ALL of the following are TRUE: 
                 - the mean radius is greater than 14.270505, 
                 - AND the perimeter error is greater than 3.346206, 
                 - AND the mean concavity is greater than 0.3000404

In the case above, a Doctor using this model might review the
proposed classification and its reasoning and decide if the
logic soundly classifies this specific patient or identify specific
aspects of the case to review more deeply.

## Use Case 3: Domain Knowledge and Control

In some applications, one might have Domain Knowledge, such as
Predictive Rules, an existing Knowledge Base, or required constraints.
NRNs can leverage this pre-existing logic, along with training data
to either improve a model's performance, expand its capabilities beyond
those possibly from supervised training only, or control its behavior
to meet pre-defined criteria.

!!! Example "Classifying Flowers"
     - A flower is in the setosa class because:
         - ALL of the following are TRUE: 
             - ANY of the following are TRUE: 
                 - NOT the following: 
                     - petal length (cm) greater than 0.95, 
                 - OR NOT the following: 
                     - sepal length (cm) greater than 0.969, 
                 - OR NOT the following: 
                     - sepal width (cm) greater than 0.078, 
                 - OR sepal length (cm) NOT greater than 0.146, 
             - AND ANY of the following are TRUE: 
                 - petal length (cm) NOT greater than 0.164, 
                 - OR sepal width (cm) greater than 0.885

In the example above, the logic below was encoded before training
the NRN, and is a part of the logic used to classify the Setotsa
flowers.

!!! Example "Our Domain Knowledge"
    - ANY of the following are TRUE:
          -  petal length (cm) NOT greater than 0.164, 
          - OR sepal width (cm) greater than 0.885

This principle can be extended to many more complex, and useful
applications.