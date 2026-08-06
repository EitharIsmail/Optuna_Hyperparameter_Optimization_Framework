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

**0.0: What is model hyperparameter and how they differ from model parameters:**
In any model there are two types of variables, parameter which are internal configuration variables that the model learn directly from the training data during training, while hyperparameters are external configuration variables whose values are set manually before the training process begins and it dedicate how the model learns. The following table shows the key differences between them:

```
| Feature                          | Parameters                      | Hyperparameter |
|----------------------------------|---------------------------------|----------------|
| Origin                           | learned automatically from data | Set manually by user or engine |
| Determination Time               | During training                 | Before Training |
| Can they change during training? | Yes, Updates continuously       | Usually remain fixed |
| Examples                         | Weights , Biases                | Learning rate, batch size, number of epochs, layer depth |
```

**0.1: What is hyperparameters tuning?**
Is the process of finding the best set of configuration settings for a machine learning model before training it to maximize its performance or to make it perform well on unseen data.

- Why we need hyperparameters tuning?
  Because different hyperparameters values dramatically affect model's performance, so we need to find the set values that make the model perform as we want before training it.
  Here is an example that shows how different learning values can affect the model performance:
  If the learning rate is too high: The model might overshoot the optimal solution and never converge.
  If the learning rate is too low: The model will take forever to train and might get stuck in a sub-optimal trap (Local Minimum).


  
