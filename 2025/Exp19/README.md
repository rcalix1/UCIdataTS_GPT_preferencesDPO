## Exp 19

* main data set used: Appliances 
* https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

## UCI data sets

* https://archive.ics.uci.edu/dataset/360/air+quality

* https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

* https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations

## Hyperparameter Recommendations for GRPO and DPO

| Method | Beta Range | Learning Rate | Notes                                          |
|--------|------------|----------------|------------------------------------------------|
| GRPO   |   0.1 – 0.5 - 2.0  | 1e-4 – 5e-4     | Try lowering β and increasing LR if too stable |
| DPO    | 0.001 - 0.05 – 0.3 | 1e-5 – 5e-5     | Too high β saturates gradients                 |

