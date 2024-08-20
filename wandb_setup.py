import wandb

# Initialize WandB
wandb.init(project="bert-naive-bayes-intent-classification")

# Log the accuracy to WandB
wandb.log({"accuracy": accuracy})

# Optionally, log the model parameters, gradients, etc.
wandb.watch(nb_classifier, log="all")
