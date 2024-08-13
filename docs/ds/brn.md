# R-Reinforced Neural Reasoning Network (R-NRN)

The Neural Reasoning Networks (NRN) alone provides powerful building blocks for 
a learning algorithm, but if we don't already have expert knowledge then
how do we leverage the Reasoning Networks logical nodes to make predictions? To do this, 
we need to mine the logic from data.  This is exactly what other 
ML/AI algorithms do.  A tree in a Random Forest model can be constructed by hand, 
but who has the time to manually build hundreds of decision trees that will 
be useful for making predictions.  R-NRN is the starting point for 
learning what logic is useful, directly from data.

The ideas for the R-NRN algorithm draw heavily from [genetic
algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm). The gist of R-NRN is to fix a logical structure
for a Neural Reasoning Network apriori, and use 
[Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning),
in this case a Multi-Armed Bandit, to efficiently search a large hypothesis space
for the useful subsets using a form of "natural selection".  
In the case of R-NRN the process is as follows:

!!! info "R-NRN Algorithm"
    1. Initialize a RN with a static logical structure.
    2. Generate logic by selecting predicates (features) for each input to the apriori logical structure drawn from a Multi-Armed Bandit policy initialized with a uniform distribution.
    3. Use the generated logic to make predictions with the RN and perform gradient descent to adjust the weights on that logic such that useful logics get higher weights and useless logic gets lower weights.
    4. Prune/Kill "useless logic", which can be identified as logic with weights below a threshold.  Keep/Let Survive logic with weights above that threshold.
    5. Update a Bayesian Multi-Armed Bandit policy over predicates (features) with rewards equal to the weights of un-pruned predicates (features).
    6. Repeat steps 2-5, generating new logic using the Bandit's policy until a stopping criteria is met.

As the algorithm trains, the set of logic that is not pruned/killed becomes more
and more useful.  Ultimately, the surviving logic can be used to make useful
predictions.

R-NRN has a specific structure of logic, such that the space of logic to 
explore is constrained to the subset that can be created using this structure.  
The structure of logic in this algorith is constructed from alternating
conjunction and disjunction `logical layers`.  We begin and end our RN with
conjunctions.

### Conjunction Layer

A conjunction layer creates tensor of conjunction (AND) operations. The
result is a logical statement such as:

!!! example "Conjunction Layer"
    - `AND(PREDICATE_1, PREDICATE_2), AND(PREDICATE_1, PREDICATE_3), AND(PREDICATE_4, PREDICATE_5)`

The number of inputs in each logical node is a hyper-parameter.  
`Conjuction Layers` can also take as input previously generated logic, 
thus making the RN deeper.

### Disjunction Layer

A disjunction layer creates a tensor of disjunctions (OR) operations. The
result is a logical statement such as:

!!! example "Disjunction Logic"
    - `OR(PREDICATE_1, PREDICATE_2), OR(PREDICATE_1, PREDICATE_3), OR(PREDICATE_4, PREDICATE_5)`

The number of inputs in each logical node is a hyper-parameter.  
`Disjuction Layers` can also take as input previously generated logic, 
thus making the RN deeper.

### Output Layer

The output layer is a conjunction of all the previous logic.
The truth value of the output layer is used as the prediction for the
classification or regression and is in the range `[0.0, 1.0]`.  One would
apply inverse transformation to this value for regression problems.
For example:

!!! example "Output Layer Logic"
    - `ClASS PREDICTION = AND(OR(AND(PREDICATE_1, PREDICATE_2), AND(PREDICATE_3, PREDICATE_4)), OR(AND(PREDICATE_4, PREDICATE_5), AND(PREDICATE_6, PREDICATE_7)))`

Even though the initial structure of logic cannot be changed, the weights that
indicated the importance of each logic will update during training.  Thus, we
can encounter entirely different logical structures when inspecting a 
RN trained by R-NRN and excluding unimportant logic, i.e. logic
with weights below a pre-determined threshold of importance. For example:

!!! example "Output Layer Logic: Thresholded by Importance"
    - `ClASS PREDICTION = AND(OR(AND(PREDICATE_1, PREDICATE_2), PREDICATE_3), AND(PREDICATE_6, PREDICATE_7))`

## Bandit Reinforced-NRN Hyper-Parameter Setting and Tuning

Through our experiments thus far we have determined that some hyper-parameters
are more important to R-NRN model performance.  The full list of hyper-parameters
is:

!!! info "R-NRN Hyper-Parameters"
    - **layer_sizes (list)**: A list containing the number of output logics for each layer.
    - **n_selected_features_input (int)**: The number of features to include in each logic in the input layer.
    - **n_selected_features_internal (int)**: The number of logics to include in each logic in the internal layers.
    - **n_selected_features_output (int)**: The number of logics to include in each logic in the output layer.
    - **perform_prune_quantile (float)**: The quantile to use for pruning RN.
    - **ucb_scale (float)**: Scale parameter in bayesian ucb policy. Controls greediness of the bandit.
    - **normal_form (str)**: One of 'cnf' for Conjunctive Normal Form, or 'dnf' for Disjunctive Normal Form.  'cnf' models alternate OR-AND logic; 'dnf' models alternate AND-OR logic.
    - **prune_strategy (str)**: One of 'class', 'logic_class', or 'logic'.  'class' pruning strategy evaluates policy updates and logic pruning on a per class (per output) basis using predicate weights.  The 'logic_class' strategy evaluates each input logic by class based on its weights. The 'logic' evaluates policy updates and logic pruning using per class per atomic-logic activations using an evaluation metric such as AUC or MSE to assess individual logic performance.
    - **delta (float)**: Controls diversity of learned logic by down-weighting the likelihood of drawing un-pruned predicates during new logic generation. Higher values increase diversity of logic generation away from existing logics.
    - **bootstrap (bool)**: Only used for 'logic' pruning strategy.  If true, each logic is independently evaluated using a bootstrap sample.
    - **swa (bool)**: If true, use Stochastic Weight Averaging to regularize the model.
    - **add_negations (bool)**: If true, logic will be generated to include both a positive and negative predicate, e.g. AND(X, NOT(X)), where the bounds are equidistint from zero
    - **weight_init (float)**: Upper bound of uniform weight initialization.  Lower bound is negated value.

!!! info "Boosted-R-NRN Hyper-Parameters (addition to R-NRN hyper-parameters)"
    - **xgb_max_depth (int)**: Max depth for XGBoost boosting model.
    - **xgb_n_estimators (int)**: Number of estimators for XGBoost boosting model.
    - **xgb_min_child_weight (float)**: Minimum child weight for XGBoost boosting model.
    - **xgb_subsample (float)**: Subsample percentage for XGBoost boosting model.
    - **xgb_learning_rate (float)**: Learning rate for XGBoost boosting model.
    - **xgb_colsample_bylevel (float)**: Column subsample percent for XGBoost boosting model.
    - **xgb_colsample_bytree (float)**: Tree subsample percent for XGBoost boosting model.
    - **xgb_gamma (float)**: Gamma parameter for XGBoost boosting model.
    - **xgb_reg_lambda (float)**: Lambda regularization parameter for XGBoost boosting model.
    - **xgb_reg_alpha (float)**: Alpha regularization parameter for XGBoost boosting model.

!!! info "BanditNRNTrainer Hyper-Parameters"
    - **model (RN)**: An instantiated RN from torchlogic.
    - **loss_func (pytorch Loss Function)**: The pytorch loss function you will use.
    - **optimizer (pytorh Optimizer)**: The pytorch optimizer you will use.  AdamW is a typical choice.
    - **scheduler (pytorh Scheduler)**: The pytorch scheduler you will use.  CosineAnnealingWarmRestarts is a typical choice.
    - **epochs (int)**: The number of epochs to use for training.
    - **learning_rate (float)**: The learning rate to use for training.
    - **accumulation_steps (int)**: The number of training steps to accumulate gradients.  Useful for large data or models that cannot fit full batches into memory.
    - **l1_lambda (float)**: The coefficient for L1 regularization.
    - **lookahead_steps (int)**: Number of steps to use in Lookahead optimization.
    - **lookahead_steps_size (int)**: Size of steps to use in Lookahead optimization.
    - **augment (str)**: One of 'MU' (MixUp), 'CM' (CutMix), 'AT' (FGSM) or None.
    - **augment_alpha (float)**: Single parameter used in data augmentation techniques.  Must be in [0, 1].
    - **class_independent (bool)**: If true, perform pruning and policy updates independently per output. Used in most multi-output cases.
    - **early_stopping_plateau_count (int)**: Number of validations without improvement to stop training after
    - **perform_prune_plateau_count (int)**: Number of validations without improvement to prune and grow RN.
    - **increase_prune_plateau_count (int)**: An increase to the perform_prune_plateau_count triggered when increase_prune_plateau_count_plateau_count is reached.
    - **increase_prune_plateau_count_plateau_count (int)**: Number of validations without improvement to trigger an increase to the perform_prune_plateau_count.
    
!!! info "Other Hyper-Parameters not part of TorchLogic"
    - **batch_size (int)**: Size of batches to load to memory using PyTorch DataLoader.
    - **weight_decay (float)**: Weight decay for optimizer.
    - **T_0 (int)**: Number of iterations for the first restart for CosineAnnealingWarmRestarts scheduler.
    - **T_mult (int)**: A factor increases T_i after a restart for CosineAnnealingWarmRestarts scheduler.

R-NRN hyper-parameter general guidelines:

| Hyper-parameter                                | Lower   | Upper            | Notes                                                                                                                                                                                                                                                                   |
|------------------------------------------------|---------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **layer_sizes**                                | [2,]    | [30, 30, 30, 30] | The choice of `layer_sizes` is highly dependent on your problem.  Higher values increase the space of random logic to search without repeating logic so long as the layer sizes are small enough for your data.                                                         |
| **n_selected_features_input**                  | 2       | 15               | The choice of `n_selected_features_input` is highly dependent on your data.  Generally, this should be set to a value less than the feature count.                                                                                                                      |
| **n_selected_features_internal**               | 2       | 10               | The choice of `n_selected_features_internal` depends on the input layer size.  Generally, this should be set to a value less than the input layer size.                                                                                                                 |
| **n_selected_features_output**                 | 2       | 10               | `n_selected_features_output` strongly controls the complexity of the model and also the explanations. For explainable models this should be set in a range that would aid interpretation.                                                                               |
| **perform_prune_quantile**                     | 0.05    | 0.9              | `perform_prune_quantile` determines how useful the surviving logic must be in a relative sense.                                                                                                                                                                         |
| **ucb_scale**                                  | 1.0     | 2.0              | `ucb_scale` the scale of the confidence interval in the multi-armed bandit policy.  c = 1.96 is a 95% confidence interval. Smaller values cause the bandit to converge quickly using a more greedy search.                                                              |
| **delta**                                      | 1.0     | 12.0             | `delta` larger values will de-correlate the atomic-level logic by biasing the logic to use different features.                                                                                                                                                          |
| **weight_init**                                | 0.05    | 1.0              | `weight_init` will change the scale of initialized weights.  Typically this should be between zero and 1, and the optimal setting is dependent on each data set.                                                                                                        |
| **learning_rate**                              | 0.0001  | 0.15             | `learning_rate` learning rate for optimization. This is highly dependent on the optimizer but for AdamW the learning rate can be fairly high.                                                                                                                           |
| **l1_lambda**                                  | 0.00001 | 0.1              | `l1_lamda` L1 regularization strength.                                                                                                                                                                                                                                  |
| **weight_decay**                               | 0.00001 | 0.1              | `weight_decay` weight decay regularization strength.                                                                                                                                                                                                                    |
| **lookahead_steps**                            | 0       | 15               | `lookahead_steps` a form of regularization using lookahead optimization.  If 0, lookahead optimization is not used.                                                                                                                                                     |
| **lookahead_steps_size**                       | 0.5     | 0.8              | `lookahead_steps_size` the step size used for lookahead optimization.                                                                                                                                                                                                   |
| **augment_alpha**                              | 0.0     | 1.0              | `augment_alpha` meaning changes dependent on the type of data augmentation.  Higher values typically create more challenging data augmentation.                                                                                                                         |
| **early_stopping_plateau_count**               | 25      | 50               | `early_stopping_plateau_count` is highly dependent on the number of epochs.  Typically we give a wide margin to allow for the algorithm to improve performance during training.                                                                                         |
| **perform_prune_plateau_count**                | 1       | 8                | `perform_prune_plateau_count` controls the frequency with which new logic is generated.  Smaller values mean we produce logic more frequently, but that logic has had fewer epochs of weight updates that determine its usefulness.                                     |
| **increase_prune_plateau_count_plateau_count** | 0       | 20               | `increase_prune_plateau_count` is the number of plateau events to increase our perform_prune_plateau_count factor by when we have stopped improving.  Setting this greater than 0 allows for the network to update less frequently once good performing logic is found. |
| **increase_prune_plateau_count_plateau_count** | 10      | 30               | `increase_prune_plateau_count_plateau_count` determines when we will stop frequent updates to the logic and allow the network to optimize the identified logic for more steps.                                                                                          |
| **T_0**                                        | 2       | 10               | `T_0` determines when we will perform a warm restart for the first time if using CosineAnnealingWithWarmRestarts scheduler.  This keesp the learning rate fairly high throughout training.                                                                              |
| **T_mult**                                     | 1       | 5                | `T_mult` determines the factor to increase the period between warm restarts.  Both scheduler parameters are dependent on the number of epochs.                                                                                                                          |
