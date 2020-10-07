import os
import shutil
import random
# Train, Validation 데이터셋 만들기

cwd = os.getcwd()

# data 폴더에 원하는 데이터를 형식에 맞추어 저장
DATA_PATH = os.path.join(cwd, 'data')
class_names = os.listdir(DATA_PATH)
base_dir = DATA_PATH



def create_training_data(folder_name):
    train_dir = f"{base_dir}/train/{folder_name}"
    files = os.listdir(os.path.join(base_dir, folder_name))

    shutil.move(f'{base_dir}/{folder_name}', train_dir)
    print("%s Complete!!" % folder_name)


# Create Training Folder
for label in class_names:
    create_training_data(label)

# Move images randomly from training to val folders
for label in class_names:
    val_folder_name = os.path.join(base_dir, 'val', label)
    try:
        os.makedirs(val_folder_name)
    except FileExistsError:
        print("Folder already exist")

train_dir = base_dir + '/train'
val_dir = base_dir + '/val'

# Training 데이터의 2할을 Validation으로 나누어 저장
for label in class_names:
    label_train = os.path.join(train_dir, label)
    label_files = os.listdir(label_train)
    for f in label_files:
        if random.random() > 0.80:
            shutil.move(f'{label_train}/{f}', f'{val_dir}/{label}')

for label in os.listdir(val_dir):
    val_label_dir = os.path.join(val_dir, label)
    print("Number of Validation", label, "images: ", len(os.listdir(val_label_dir)))

