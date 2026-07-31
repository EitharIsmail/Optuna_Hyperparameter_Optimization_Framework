# Optuna_Hyperparameter_Optimization_Framework
📘 How to Use This Notebook
This is a standalone, self-teaching lecture notebook.

How to Study:

Read all markdown cells carefully - they contain explanations, mental models, and reasoning
Run code cells sequentially - don't skip ahead as each builds on the previous
Pause at "Stop & Think" prompts - make predictions before running the next cell
Experiment with the code - change parameters and see what happens
Time Commitment: ~(    ) hours for complete understanding

Learning Outcomes: After completing this notebook, you will be able to:

Explain what is Optuna, why it was create, what problems it solve and how.
Use Optuna


# Part 0: Hyperparameter Tuning - The problem

In any model there are two types of variables, parameter which are internal configuration variables that the model learn directly from the training data during training, while hyperparameters are external configuration variables whose values are set manually before the training process begins. The following table shows the key differences between them:

```
| Feature                          | Parameters                      | Hyperparameter |
|----------------------------------|---------------------------------|----------------|
| Origin                           | learned automatically from data | Set manually by user or engine |
| Determination Time               | During training                 | Before Training |
| Can they change during training? | Yes, Updates continuously       | Usually remain fixed |
| Examples                         | Weights , Biases                | Learning rate, batch size, number of epochs, layer depth |
```

