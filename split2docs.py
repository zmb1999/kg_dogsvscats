import os
import shutil
from tqdm import tqdm

def split2docs():
    os.chdir(r"D:\\machine_learning\\dogvscat_data\\")

    train_filenames = os.listdir('train')

    train_cat = filter(lambda x: x[:3] == 'cat', train_filenames)
    train_dog = filter(lambda x: x[:3] == 'dog', train_filenames)

    def rmrf_mkdir(dirname):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)

    rmrf_mkdir('train2')
    os.makedirs('train2/cat')
    os.makedirs('train2/dog')

    rmrf_mkdir('test2')
    os.makedirs('test2/test')

    # print(os.getcwd())
    pbar = tqdm(os.listdir('test'))
    for filename in pbar:
        pbar.set_description("Processing %s" % filename)
        shutil.copy('test/' + filename, 'test2/test/' + filename)

    pbar = tqdm(list(train_cat))
    for filename in pbar:
        pbar.set_description("Processing %s" % filename)
        shutil.copy('train/' + filename, 'train2/cat/' + filename)

    pbar = tqdm(list(train_dog))
    for filename in pbar:
        pbar.set_description("Processing %s" % filename)
        shutil.copy('train/' + filename, 'train2/dog/' + filename)



if __name__ == '__main__':

    split2docs()
