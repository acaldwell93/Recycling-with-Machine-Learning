import os
import numpy as np

train_filepath='../data/images/train'
holdout_filepath='../data/images/holdout'

folders = ['paper', 'metal', 'cardboard', 'trash', 'glass', 'plastic']

for cat in folders:
    cat_lst = os.listdir(os.path.join(train_filepath, cat))
    test_lst = np.random.choice(cat_lst, size=int(.1*len(cat_lst)), replace=False)
    for img in test_lst:
        current_path = os.path.join(train_filepath, cat, img)
        new_path = os.path.join(holdout_filepath, cat, img)
        os.rename(current_path, new_path)