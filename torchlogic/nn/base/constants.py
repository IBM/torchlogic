keyword_constraints = ['scenarios', 'requirements', 'conditions', 'situations', 'circumstances', 'context']

or_options = ["There are several scenarios that could be met for this prediction to hold true.  "
              "The first scenario that could be met is as follows.  ",
              "At least one of the following requirements are met.  The first requirement is as follows. ",
              "There are several more conditions that could be met.  The conditions are described next.  ",
              "In fact, there are additional situations that could be met.  ",
              "It must be true that at least one of the following circumstances are met.  "
              "The first circumstance is the following.  ",
              "There is additional context that could be met.  The contexts are described next.  "]
and_options = ["There are several scenarios that must be met for this prediction to hold true.  "
               "The first scenario is as follows.  ",
               "All of the following requirements are met. The first requirement is as follows.  ",
               "There are several more conditions that must be met.  The conditions are described next.  ",
               "In fact, there are additional situations that must be met.  ",
               "It must be true that the following circumstances are met.  The first circumstance is the following.  ",
               "There is additional context that must be met.  The contexts are described next.  "]

or_options_negated = ["There are several scenarios that must NOT be met for this prediction to hold true. "
                      "The first scenario that must NOT be met is as follows.  ",
                      "All of the following requirements are NOT met. The first requirement is as follows.  ",
                      "There are several more conditions that must NOT be met.  The conditions are described next.  ",
                      "In fact, there are additional situations that must NOT be met.  "
                      "The first situation is as follows. ",
                      "It must be true that all of the following circumstances are NOT met.  "
                      "The first circumstance is the following.  ",
                      "There is additional context that must NOT be met.  The contexts are described next.  "]
and_options_negated = ["There are several scenarios, at least one of which must NOT hold,"
                       " for this prediction to hold true. "
                       "The first scenario is as follows.  ",
                       "It must be true that at least one of the "
                       "following requirements are NOT met.  The first requirement is as follows. ",
                       "There are several more conditions, at least one of which must NOT be met.  "
                       "The conditions are described next.  ",
                       "In fact, there are additional situations, at least one of which must NOT be met.  ",
                       "It must be true that at least one of the following circumstances are NOT met.  "
                       "The first circumstance is the following.  ",
                       "There is additional context, at least one of which must NOT be met.  "
                       "The contexts are described next.  "]

and_joining_options = [".\n\nThe next scenario that must be met is as follows.  ",
                       ".  An additional requirement that must be met is the following.  ",
                       ".  As well as the following conditions.  ",
                       ".\n\nThe next situation that must be met is as follows.  ",
                       ".  An additional circumstance that must be met is the following.  ",
                       ".  As well as the following context.  "]
or_joining_options = [".\n\nThe next scenario that could be met is as follows.  ",
                      ".  An additional requirement that could be met is the following.  ",
                      ".  Or the following conditions.  ",
                      ".\n\nThe next situation that could be met is as follows.  ",
                      ".  An additional circumstance that could be met is the following.  ",
                      ".  Or the following context.  "]


and_joining_options_negated = [".\n\nThe next scenario that could NOT be met is as follows.  ",
                               ".  An additional requirement that could NOT be met is the following.  ",
                               ".  Or the following conditions are NOT met.  ",
                               ".\n\nThe next situation that coule NOT be met is as follows.  ",
                               ".  An additional circumstance that could NOT be be met is the following.  ",
                               ".  Or the following context is NOT met.  "]
or_joining_options_negated = [".\n\nThe next scenario that must NOT be met is as follows.  ",
                              ".  An additional requirement that must NOT be met is the following.  ",
                              ".  The following conditions are also NOT met.  ",
                              ".\n\nThe next situation that must NOT be met is as follows.  ",
                              ".  An additional circumstance that must NOT be met is the following.  ",
                              ".  And the following context is NOT met.  "]