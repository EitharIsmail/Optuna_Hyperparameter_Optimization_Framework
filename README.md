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

Why we need hyperparameters tuning?

  Because different hyperparameters values dramatically affect model's performance (accuracy, stability, and generalization), so we need to find the set values that make the model perform as we want before training it.
  Here is an example that shows how different learning values can affect the model performance:
  If the learning rate is too high: The model might overshoot the optimal solution and never converge.
  If the learning rate is too low: The model will take forever to train and might get stuck in a sub-optimal trap (Local Minimum).

**0.2: Traditional (Classical) techniques used to perform hyperparameters**

**0.2.1: Manual Search (Oldest Technique):**

This is the earliest and simplest approach. The machine learning engineer chooses hyperparameters based on experience, trains the model, observes the results, and adjusts them iteratively. This process relies entirely on human intuition.

Advantages:
- Simple.
- No additional tools required.
- Helps beginners understand the effect of each hyperparameter.
  
Disadvantages:
- Time-consuming.
- Depends heavily on experience.
- Can easily miss the best hyperparameter values.

**0.2.2: Grid Search (Traditional Standard):**

Grid search systematically test all the combinations for all the possible values for all the hyperparameters. So we define a fixed grid of values for each hyperparameter, and the algorithm tests every single possible combination (the Cartesian product).

For example is we want to tune the following:

Learning Rate = [0.1, 0.01, 0.001]

Batch Size = [16, 32, 64]

Optimizer = [SGD, Adam]

The Grid Search tests every combination:

```
|Trial |	Learning Rate | Batch Size | Optimizer |
|------|----------------|------------|-----------|
| 1	   | 0.1	          | 16	       | SGD       |
| 2	   | 0.1	          | 16	       | Adam      |
| 3	   | 0.1	          | 32	       | SGD       |
| ...	 | ...	          | ...	       | ...       |
| 18	 | 0.001	        | 64	       | Adam      |
```
Advantages:

- Very easy to implement.
- Guaranteed to test every specified combination.
- Good when there are only a few hyperparameters.

Disadvantages:
