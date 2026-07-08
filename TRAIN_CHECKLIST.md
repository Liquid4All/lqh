# Training checklist

For the training skill we need to make sure the agent (= orchestration model) has the following context in the skill.md file:
Below outlines the ideal or typical training process that the agent should follow by default. If the user explicitly mentions or directs to skip steps or jumps to ahead then do so but with a warning that this is not the ideal process and may lead to suboptimal results.


- Prerequisite: Before starting any training run, the agent must have ensure that the data quality is good and the scorer/judging quality is good. 
This means the agent has itself read the example (demo) data as well as the scoring output. 
If the generated data or scoring is poor quality, the agent need to iterate on it as outlined in the data gen skill.
- Before using any data we need to make sure they are properly filtered using the appropriate tool that runs the scorer over them
- Once we have a good data and scoring, we need to create + filter an evaluation (= validation) dataset, typically a few hundred examples, that is not used in training but only for evaluation.
- Next we should do a zero-shot evaluation of the base model on the evaluation dataset, to have a baseline score to compare to. VERY IMPORTANT: Make sure that the agent uses an appropriate system prompt! In the past it has been a typical common mistake by the agent to leave out the system prompt which leads to poor results due to a confused model, ie. does not know what to do. Make sure in the training skill and general system prompt (or tool definition of the zero shot eval) that we should include a well structured system prompt with clear instructions, expected output format, and ideally examples.
- After that, we generate a first small training dataset, of a few hundred to low thousands of examples, and train a first model on it using SFT. This can be done with default hyperparameters, no need for hyperparameter tuning at this stage.
- After the training we evaluate the model on the evaluation dataset + scoring. If we see the direction is right, we can then generate a larger training dataset, of tens of thousands of examples. If the results is a degradation we should try to do hyperparameter tuning, or if that doesn't work we should try to improve the data quality and scoring quality.
In case of a unchanged score, we can proceed also with scaling the dataset but with caution, as it might be a sign of overfitting or underfitting.
- Once we have a larger dataset, we should do some hyperparameter tuning, to find the best learning rate, batch size, number of epochs, etc.
- We then evaluate and score the best model.
- If we see an improvement, we can continue to scale the dataset further, and if we see a degradation we should try to do hyperparameter tuning, or if that doesn't work we should try to improve the data quality and scoring quality.
- Once we are stuck, or deal with a high average but outlier, we can look into DPO preference alignment training. 
- The base model for DPO should be the best SFT model we have.
- Typically, DPO is very sensitive to hyperparameters as well as the start of DPO we already have a large SFT dataset, thus we can directly go to hyperparameter tuning for DPO, without doing a first run with default hyperparameters.
- We then score the best DPO model and compare to the best SFT and baseline models.


Common failure modes and issues:
Keep a section in the training skill that list common issues and failure modes, such as:
- Not inspecting the data and scoring quality before starting training, which can lead to poor results and wasted compute.
- Forgetting the system prompt in the zero-shot evaluation, which can lead to a very low baseline score and make it hard to see improvements.
- Either staying at a very small dataset or immediately scaling dataset without validating the direction, ie. we need at least a few thousand training samples before we can expect to see improvements.
- Not filtering the dataset but directly training on the raw generated data, which can lead to poor results due to low quality data and scoring.
- Not doing hyperparameter tuning, which can lead to suboptimal results and not seeing improvements even
- Directly jumpting to DPO without doing a thorough SFT training and hyperparameter tuning and scaling the SFT dataset.
- Using a too small base model, eg. we cannot expect LFM2.5-350M to solve a task that requires a model 1000x in size. If the model performs poorly even when trained on a large high quality dataset, we may want to test a larger base model.

