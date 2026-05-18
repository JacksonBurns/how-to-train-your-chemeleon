# for data preparation
WINSORIZATION_FACTOR = 3

# data loading during training - keeps columns above this threshold
PERCENTILE_THRESHOLD = 0.30

# for training
DROPOUT_FRACTION = 0.85  # see https://doi.org/10.1039/D5DD00369E for better ideas -> 60% mask = 40% dropout
EPOCHS = 10
PATIENCE = 2
LR_MULTIPLIER = 2.8  # sqrt(8), from: https://arxiv.org/pdf/1705.08741
INITIAL_LEARNING_RATE = 0.0001 * LR_MULTIPLIER
MAXIMUM_LEARNING_RATE = 0.001 * LR_MULTIPLIER
FINAL_LEARNING_RATE = 0.0001 * LR_MULTIPLIER
WARMUP_EPOCHS = 1
CHUNKS_PER_BATCH = 2

# model hyperparameters
FNN_HIDDEN_SIZE = 1_024
FNN_HIDDEN_LAYERS = 1
FNN_ACTIVATION = "RELU"  # one of: RELU, LEAKYRELU, PRELU, TANH, ELU
MP_HIDDEN_SIZE = 2_048
MP_DEPTH = 6
MP_ACTIVATION = "RELU"  # one of: RELU, LEAKYRELU, PRELU, TANH, ELU
FEATURIZER = "V2"  # one of: "V2", "RIGR"
MP_TYPE = "DEFAULT"  # one of: "DEFAULT", "UNTIED"
