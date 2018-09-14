from __future__ import print_function

import SharedArray as sa
import numpy as np
import os
from sklearn.utils import shuffle



def save_on_sa(data_dir, use_only_84_keys = True, rescale = True, postfix=''):
    print('Reading...')
    print('[*]',data_dir)

    ##data_prefix = ['Bass', 'Drum', 'Guitar', 'Other', 'Piano', 'Chord']
    data_prefix = ['mel_phr','acc_phr']
    subdirs = ['tra', 'val']

    for sd in subdirs:
        data = []
        # lead sheet setting
        for dp in range (2):
            x_name = data_prefix[dp]
            print (os.path.join(data_dir, sd , x_name+'.npy'))
            ##tmp_data =  np.reshape(np.load(os.path.join(data_dir, sd , x_name+'.npy')),(-1,384,128, 1)) # for 4 dbar
            tmp_data =  np.reshape(np.load(os.path.join(data_dir, sd , x_name+'.npy')),(-1,96,128, 1)) # for 1bar
            if(use_only_84_keys):
                tmp_data = tmp_data[:, :, 24:108, :]
            data.append(tmp_data)
        
        data_X = np.concatenate(data,axis = 3)
        
        # midi setting
        ##data_X = np.load(os.path.join(data_dir, sd, 'phr_chord_clean.npy'))
        
        
        ##data_y = np.load(os.path.join(data_dir, sd ,  data_prefix[-1]+'.npy'))

        print(data_X.dtype)
        ##print(data_y.dtype)
        if sd is 'tra':
            print(sd)
            print('Shuffling...')
            ##data_X, data_y = shuffle(data_X, data_y, random_state=0)
            data_X = shuffle(data_X, random_state=0)
        else:
            print(sd)
            pass
        name = sd + '_X_' + postfix
        print(name, data_X.shape)
        tmp_arr = sa.create(name, data_X.shape, dtype=bool)
        np.copyto(tmp_arr, data_X)

        ##name = sd + '_y_' + postfix
        ##print(name, data_y.shape)
        ##tmp_arr = sa.create(name, data_y.shape, dtype=bool)
        ##np.copyto(tmp_arr, data_y)

if __name__ == '__main__':
    
    ##save_on_sa('../../wayne/v3.0/dataset/data_bar', postfix='bars')
    ##save_on_sa('../music/lpd_4dbar_12_C', postfix='phrs')  # data_phr
    ##save_on_sa('./data_tab_4dbar_12', postfix='phrs')
    ##save_on_sa('./data_tab_1bar_12', postfix='phrs')
    save_on_sa('./data_tab_2bar_12', postfix='phrs')