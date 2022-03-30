import os

def splits(splits_dir, data_dir, ratio=0.2):

    """
    Args:
        splits_dir : Splits text file "~/splits"
        data_dir : Images directory "~/Images"
        ratio : Validate datasets ratio (default=0.2)
    """
    os.mkdir(splits_dir)

    if not os.path.exists(data_dir):
        raise FileNotFoundError

    ims = os.listdir(data_dir)
    t_list = []
    v_list = []

    for i in range(len(ims)):
        if i / len(ims) < ratio:
            with open(os.path.join(splits_dir, 'val.txt'), "a") as f:
                f.write(ims[i].split('.')[0] + '\n')
            v_list.append(ims[i].split('.')[0])
        else:
            with open(os.path.join(splits_dir, 'train.txt'), "a") as f:
                f.write(ims[i].split('.')[0] + '\n')
            t_list.append(ims[i].split('.')[0])