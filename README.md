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

### 0.0: What is model hyperparameter and how they differ from model parameters:

  In any model there are two types of variables, parameter which are internal configuration variables that the model learn directly from the training data during training, while hyperparameters are external configuration variables whose values are set manually before the training process begins and it dedicate how the model learns. The following table shows the key differences between them:

```
| Feature                          | Parameters                      | Hyperparameter                                           |
|----------------------------------|---------------------------------|----------------------------------------------------------|
| Origin                           | learned automatically from data | Set manually by user or engine                           |
| Determination Time               | During training                 | Before Training                                          |
| Can they change during training? | Yes, Updates continuously       | Usually remain fixed                                     |
| Examples                         | Weights , Biases                | Learning rate, batch size, number of epochs, layer depth |
```

### 0.1: What is hyperparameters tuning?

  Is the process of finding the best set of configuration settings for a machine learning model before training it to maximize its performance or to make it perform well on unseen data.

Why we need hyperparameters tuning?

  Because different hyperparameters values dramatically affect model's performance (accuracy, stability, and generalization), so we need to find the set values that make the model perform as we want before training it.
  
  Here is an example that shows how different learning values can affect the model performance:
  If the learning rate is too high: The model might overshoot the optimal solution and never converge.
  If the learning rate is too low: The model will take forever to train and might get stuck in a sub-optimal trap (Local Minimum).

### 0.2: Early techniques used to perform hyperparameters:

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
| 18	 | 0.001 	        | 64	       | Adam      |

```
Advantages:

- Very easy to implement.
- Guaranteed to test every specified combination.
- Good when there are only a few hyperparameters.

Disadvantages:

- It suffers heavily from combinatorial explosion. Adding parameters or finer value increments causes the number of required training runs to skyrocket into thousands or millions.
  
  For example is we have 6 hyperparameters, and each has 10 values, then the grid search must perform 10⁶ = 1,000,000.

**0.2.3: Random Search (Blind Sampling):**

Instead of checking every point on a rigid grid, the engineer define statistical distributions (or a pool of choices) for each parameter, and the algorithm randomly selects a fixed number of combinations to test. The main problem in this search is that there is no memory from previous trails about what went right or wrong so the algorithm may repeat it again, this is called blind search.

Advantages:

- Much faster than Grid Search.
- Explores more of the search space.
- Works well in high-dimensional problems.

Disadvantages:

- Still random.
- May miss the optimum.
- Does not learn from previous trials.

**0.2.4: Bayesian Optimization:**

Instead of randomly selecting hyperparameters, this techniques learns from previous experiments. The idea is to build a probabilistic model (surrogate model) of the objective function and use this model to select the most promising points to evaluate next. This approach is particularly useful in scenarios where the objective function is unknown, noisy, or costly to evaluate, as it aims to minimize the number of evaluations required to find the optimal solution.

- What is the Surrogate model?

The real objective function might be expensive in term of time or computational power:

```
hyperparameters → train model → validation accuracy
```
For example:
```
learning_rate = 0.001
batch_size = 64
       ↓
Train neural network
       ↓
accuracy = 87%
```
Training the model could take 30 minutes.

Instead of repeatedly training the real model, Bayesian Optimization builds a cheap approximation, called a surrogate model. It tries to learn: 
```
hyperparameters → expected performance
```
The Gaussian Processes are used as surrogate model, they are probabilistic machine learning models used for regression and classification tasks. The main difference from regular models is the instead of predicting a single value, they provide both predictions and a measure of uncertainty, making them useful for problems where confidence in predictions is important.

Advantages:

- Requires far fewer trials.
- Excellent for expensive models.
- Widely used in research and industry.
  
Disadvantages:

- More mathematically complex.
- Some overhead in modeling the search process.
- Can be less effective in extremely high-dimensional spaces.

### 0.3: The need for Optuna:

Optuna sits between brute‑force and full Bayesian frameworks. It provides intelligent samplers (e.g., TPE, CMA‑ES) that learn from past trials, supports pruning (stop bad trials early), and allows expressive, conditional search spaces defined inside the objective function.

Let us see how Optuna tackled the problems faced by previous techniques:

Optuna use a method called Tree-Structured Parzen Estimator in which instead of trying every combination or randomly select them, in stead it model "good" vs "bad" trails distribution, then focusing on the good trails distribution it ask whether the hyperparameter value belongs to the distribution, instead of predicting the performance. In other words, it look if the hyperparameter look similar to the hyperparameters that have already produced good results.

- How it works:
  1. Initial Random Trials (Warm-up):
     
     Before building any statistical model, Optuna runs a few initial trials (controlled by n_startup_trials, usually set to 10) using purely random sampling. This builds an initial baseline dataset of parameters and their corresponding performance scores.
     
  2. Splitting the results (good VS bad):
 
     Once enough trials are complete, Optuna splits the history into two distinct groups based on a threshold (often determined by a percentile parameter called $\gamma$, e.g., the top 20% vs. the remaining 80%).
     - Group 1: Configurations that yielded good (low error / high accuracy) results.
     - Group 2: Configurations that yielded bad results.
     
  3. Building the Distributions (l(x) and g(x)):
  4. 
     Following the Bayesian approach, Optuna fits probability density functions (often using Gaussian Mixture Models) to both groups:
     l(x) = P(x \ y < y*): The probability distribution of parameters (x) given that the score (y) is better than the threshold (y*) (the "good" distribution).
     g(x) = P(x \ y >= y*): The probability distribution of parameters (x) for the remaining trials (the "bad" distribution).
     
  6. Choose the next trail values: The model set values and compute the ratio for them using l(x) / g(X) to see from which distribution it will come.

Optuna can be defined as a framework/tool use to orchestrate the experiment process to find the optimum set of hyperparameters values. It
```
- Suggests hyperparameters
- Runs your objective function
- Receives the result
- Records the result
- Decides what to try next
- Repeats
- Keeps the best configuration
```

Finally hyperparameter optimization has moved from brute-force (grid) to guesswork (random) to informed exploration (Optuna). For practitioners, this means less time wasted and better results, especially with models that take hours to train.

