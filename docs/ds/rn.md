# Neural Reasoning Networks (NRN)

Reasoning Networks (NRN) provides the building
blocks from which logical rules can be learned.  The current version of RN 
consists of 4 logical operators:
1. `And`
2. `Or`
3. `Not`
4. `Predicate`

!!! info "Truth value table for AND and OR"

| P   | Q   | AND | OR  |
|-----|-----|-----|-----|
| T   | T   | T   | T   |
| T   | F   | F   | T   |
| F   | T   | F   | T   |
| F   | F   | F   | F   |

These logical operations are familiar to most, except for Predicate.  Within the RN
framework, Predicates can be understood as being analogous to a column of data.  They
represent some information about the known state of the world.  For example, if we
are classifying cats and dogs the predicate `BARKS` and `MEOWS`, where the truth value
for any particular sample is `True` or `False`, would be quite useful.

### Toy Example:

For example, if we would like to classify samples as cats or dogs, one could create an RN as follows
supposing we have the data below.

| MEOWS  | BARKS | HAS FUR | IS CAT |
|--------|-------|---------|--------|
| T      | F     | T       | T      |
| T      | F     | F       | T      |
| F      | T     | T       | F      |
| T      | T     | T       | F      |

!!! example "RN Logic for Cat/Dog Classification"
    - `CLASS CAT: AND(MEOWS, NOT(BARKS))`
    - `CLASS DOG: AND(OR(MEOWS, BARKS), HAS FUR)`

The Data Scientist can construct this model directly and proceed to use it to
make predictions.  Passing our data through the logic will result in the logic
for `CLASS CAT` evaluating to `True` for cats and `False` for dogs.  `CLASS DOG`
logic will evaluate to `True` for dogs and `False` for cats.  Since this problem
is a binary classification we could use either `CLASS CAT` or `CLASS DOG` logic,
but the example shows how RN can be used for multi-class and multi-label 
problems by created logic for each class.

Each logical operation in an RN contains weights, which can be changed
using backpropogation and optimized via gradient descent to identify a
weighted logic that will minimize the loss for the classification problem.
For example, the predicate `BARKS` in our data above is perfectly correlated
to our target `IS CAT`.  Thus, optimizing the weights for the `CLASS CAT`
logic `AND(MEOWS, NOT(BARKS))` might result in the following weights for 
`MEOWS` AND `NOT(BARKS)` in the `AND` node:

!!! example "Learned RN Weights"
     - `AND(MEOWS, NOT(BARKS))`
        - `WEIGHTS: [0.5, 5.0]`
        - `BIAS: 1.0`

The relatively high weight on the `NOT(BARKS)` predicate indicates that it is
more important to the truth of the `IS CAT` logic then the `MEOWS`
predicate.  This makes sense, given that we have one instance where a
dog both barks and meows, but a cat never barks.