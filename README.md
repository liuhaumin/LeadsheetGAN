# LeadsheetGAN 自動簡譜生成 :musical_note:
[Lead Sheet GAN](https://liuhaumin.github.io/LeadsheetArrangement/) is a task to automatically generate lead sheets. There are several types we use in generation.
- **Unconditional generation:** generate melody and chords from nothing
- **Conditional generation:** generate melody-conditioned chord or chord-conditioned melody

We train the model with TheoryTab (TT) dataset to generate pop song style leadsheets.

Sample results are available
[here](https://liuhaumin.github.io/LeadsheetArrangement/results).

## Papers

__Lead sheet generation and arrangement by conditional generative adversarial network__<br>
Hao-Min Liu and Yi-Hsuan Yang,
to appear in *International Conference on Machine Learning and Applications* (ICMLA), 2018.
[[arxiv](https://arxiv.org/abs/1807.11161)]

__Lead sheet and Multi-track Piano-roll generation using MuseGAN__<br>
Hao-Min Liu, Hao-Wen Dong, Wen-Yi Hsiao and Yi-Hsuan Yang,
in *GPU Technology Conference* (GTC), 2018.
[[poster](https://liuhaumin.github.io/LeadsheetArrangement/pdf/GTC_poster_HaoMin.pdf)]

## Usage
### Step 1: adjust training or testing modes in main.py
```python
import tensorflow as tf
from musegan.core import MuseGAN
from musegan.components import NowbarHybrid
from config import *

# Initialize a tensorflow session

""" Create TensorFlow Session """
with tf.Session() as sess:
    
    # === Prerequisites ===
    # Step 1 - Initialize the training configuration        
    t_config = TrainingConfig
    t_config.exp_name = 'exps/nowbar_hybrid'        

    # Step 2 - Select the desired model
    model = NowbarHybrid(NowBarHybridConfig)
    
    # Step 3 - Initialize the input data object
    input_data = InputDataNowBarHybrid(model)
    
    # Step 4 - Load training data
    path_x_train_bar = 'tra_X_bars'
    path_y_train_bar = 'tra_y_bars'
    input_data.add_data_sa(path_x_train_bar, path_y_train_bar, 'train') # x: input, y: conditional feature
    
    # Step 5 - Initialize a museGAN object
    musegan = MuseGAN(sess, t_config, model)
    
    # === Training ===
    musegan.train(input_data)

    # === Load a Pretrained Model ===
    musegan.load(musegan.dir_ckpt)

    # === Generate Samples ===
    path_x_test_bar = 'val_X_bars'
    path_y_test_bar = 'val_y_bars'
    input_data.add_data_sa(path_x_test_bar, path_y_test_bar, key='test')
    musegan.gen_test(input_data, is_eval=True)

```
### Step 2: run store_sa.py
### Step 3: run main.py
