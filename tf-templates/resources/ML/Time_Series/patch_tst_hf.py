from transformers import PatchTSTConfig, PatchTSTModel

# Initializing an PatchTST configuration with 12 time steps for prediction
configuration = PatchTSTConfig(prediction_length=12)

# Randomly initializing a model (with random weights) from the configuration
model = PatchTSTModel(configuration)

# Accessing the model configuration
configuration = model.config