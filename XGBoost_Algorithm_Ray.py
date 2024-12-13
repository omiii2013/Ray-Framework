#!/usr/bin/env python
# coding: utf-8

pip install ray


import ray
import os
from ray.air.config import ScalingConfig
from ray.train.xgboost import XGBoostTrainer
from ray import tune
from ray.tune.tuner import Tuner, TuneConfig
from ray.air.config import RunConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
import time

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the dataset
dataset_path = os.path.join(current_dir, 'Datasets', 'breast_cancer.csv')


dataset = ray.data.read_csv(dataset_path)

train_dataset, valid_dataset = dataset.train_test_split(test_size=0.2)
test_dataset = valid_dataset.drop_columns(cols=["diagnosis"])

# NOTE: CPU does not have enough resources to run this example.
# I tried using num_workers=1, resources_per_worker={"CPU": 1, "GPU": 0} in your
# ScalingConfig below.
trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(
        num_workers=3,
        use_gpu=False,
    ),
    label_column="diagnosis",
    num_boost_round=30,
    params={
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
)
result = trainer.fit()
print(result.metrics)

# Define a custom log directory
custom_log_dir = current_dir +"\\ray_logs"

# Ensure the directory exists
import os
os.makedirs(custom_log_dir, exist_ok=True)

# Define the custom trial directory name function (if needed)
def trial_dirname_creator(trial):
    """Create shorter directory names to avoid path length issues."""
    return f"{trial.trainable_name}_{trial.trial_id}"

param_space = {"params": {"max_depth": tune.choice([1, 5, 50, 100])}}
metric = "train-logloss"

tuner = Tuner(
    trainer,
    param_space=param_space,
    run_config=RunConfig(
        verbose=1,
        callbacks=[],  # Disable all callbacks including TensorBoard
        storage_path=custom_log_dir,
    ),
    tune_config=TuneConfig(num_samples=5, metric=metric, mode="min", trial_dirname_creator=trial_dirname_creator),
)
result_grid = tuner.fit()

best_result = result_grid.get_best_result()
print("Best Result:", best_result)


best_logdir = best_result.path
print(best_logdir)
print('\n')

best_metrics = best_result.metrics
print(best_metrics)
print('\n')

best_config = best_result.config
print(best_config)


# ### XGBoost Model without Ray:

# Load the dataset
dataset = pd.read_csv(dataset_path)

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=13)

# Separate features and target for train and validation sets
X_train_all_features = train_dataset.drop(columns=["diagnosis"])
y_train = train_dataset["diagnosis"]

X_test_all_features = test_dataset.drop(columns=["diagnosis"])
y_test = test_dataset["diagnosis"]

# Define depths to test
depths = [1,5,50,100]

results = []

for max_depth in depths:
    # Create the model with different max_depth values
    model = XGBClassifier(
        max_depth=max_depth,
        n_estimators=100,
        learning_rate=0.3,
        objective='binary:logistic',
        eval_metric=['logloss', 'error']
    )

    # Start timing
    start_time = time.time()

    # Train the model
    model.fit(X_train_all_features, y_train, eval_set=[(X_test_all_features, y_test)], verbose=True)

    # End timing
    end_time = time.time()

    # Predictions
    y_pred_prob = model.predict_proba(X_test_all_features)[:, 1]  # Predicted probabilities
    y_pred = model.predict(X_test_all_features)  # Binary predictions

    # Calculate metrics
    logloss = log_loss(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)

    # Store results
    results.append({
        'max_depth': max_depth,
        'training_time': end_time - start_time,
        'log_loss': logloss,
        'accuracy': accuracy
    })

    print(f"Completed training with max_depth = {max_depth}")
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

# Output all results
for result in results:
    print(result)


# In[ ]:




