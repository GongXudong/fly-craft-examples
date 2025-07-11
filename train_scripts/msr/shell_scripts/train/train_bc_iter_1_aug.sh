#!/bin/bash


#---------------------------------------- epsilon = [1.0, 0.3, 0.3] -------------------------------------------------------------
# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_1_aug/epsilon_0_1_reg_0_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_1_aug/epsilon_0_1_reg_0_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_1_aug/epsilon_0_1_reg_0_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_1_aug/epsilon_0_1_reg_0_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_1_aug/epsilon_0_1_reg_0_N_16/128_128_seed_5.json

python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_001/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_001/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_001/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_001/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_001/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.01
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_01/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_01/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_01/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_01/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_01/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.1
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_1/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_1/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_1/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_1/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_1/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 1.0
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_1/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_1/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_1/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_1/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_1/128_128_seed_5.json



# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.0001  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_1_aug/epsilon_0_1_reg_0_0001_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_1_aug/epsilon_0_1_reg_0_0001_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_1_aug/epsilon_0_1_reg_0_0001_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_1_aug/epsilon_0_1_reg_0_0001_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/iter_1_aug/epsilon_0_1_reg_0_0001_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.001  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_001_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.01  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_01_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_01_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_01_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_01_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_01_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 0.1  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_1_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_1_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_1_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_1_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_0_1_N_16/128_128_seed_5.json

# epsilon = [1.0, 0.3, 0.3]  regularization_strength = 1.0  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_1_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_1_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_1_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_1_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_1_reg_1_N_16/128_128_seed_5.json

#---------------------------------------- epsilon = [5.0, 1.5, 1.5] -------------------------------------------------------------
# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0/128_128_seed_5.json

# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.001
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_001/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_001/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_001/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_001/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_001/128_128_seed_5.json

# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.01
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_01/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_01/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_01/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_01/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_01/128_128_seed_5.json

# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.1
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_1/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_1/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_1/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_1/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_1/128_128_seed_5.json

# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 1.0
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_1/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_1/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_1/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_1/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_1/128_128_seed_5.json


# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.001  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_001_N_16/128_128_seed_5.json

# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.01  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_01_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_01_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_01_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_01_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_01_N_16/128_128_seed_5.json

# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 0.1  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_1_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_1_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_1_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_1_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_0_1_N_16/128_128_seed_5.json

# epsilon = [5.0, 1.5, 1.5]  regularization_strength = 1.0  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_1_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_1_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_1_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_1_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_0_5_reg_1_N_16/128_128_seed_5.json

#---------------------------------------- epsilon = [10.0, 3.0, 3.0] -------------------------------------------------------------
# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0/128_128_seed_5.json

# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.001
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_001/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_001/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_001/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_001/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_001/128_128_seed_5.json

# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.01
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_01/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_01/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_01/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_01/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_01/128_128_seed_5.json

# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.1
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_1/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_1/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_1/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_1/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_1/128_128_seed_5.json

# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 1.0
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_1/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_1/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_1/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_1/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_1/128_128_seed_5.json


# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.001  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_001_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_001_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_001_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_001_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_001_N_16/128_128_seed_5.json

# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.01  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_01_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_01_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_01_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_01_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_01_N_16/128_128_seed_5.json

# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 0.1  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_1_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_1_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_1_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_1_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_0_1_N_16/128_128_seed_5.json

# epsilon = [10.0, 3.0, 3.0]  regularization_strength = 1.0  N=16
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_1_N_16/128_128_seed_1.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_1_N_16/128_128_seed_2.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_1_N_16/128_128_seed_3.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_1_N_16/128_128_seed_4.json
python train_scripts/msr/train/train_bc.py --config-file-name configs/train/msr/bc/hard/epsilon_1_reg_1_N_16/128_128_seed_5.json
