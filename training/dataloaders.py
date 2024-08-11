# Imports
import torch
import random
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms


# Define data transformations
train_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


# Define custom dataset class
class MotorDataset(Dataset):
    def __init__(self, file_list, transform=None, mask_cond=None, dataset_type=None, model_type=None):
        self.file_list = file_list
        self.transform = transform
        self.direction_map = lambda x: 'right' if x<0 else 'left'
        self.direction_encoded_map = lambda x: 1 if x<0 else 0
        self.fixation_map = None
        self.mask_cond = mask_cond
        self.dataset_type = dataset_type
        self.model_type = model_type

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    def get_opt_mag_avg(self,idx):
        img_path = self.file_list[idx]
        data_dict = pickle.load(open(img_path, "rb"))
        img = data_dict['opt_mag_avg']
        img = Image.fromarray(img)
        img_transformed = self.transform(img)
        return img_transformed

    def get_opt_ang_avg(self,idx):
        img_path = self.file_list[idx]
        data_dict = pickle.load(open(img_path, "rb"))
        img = data_dict['opt_ang_avg']
        img = Image.fromarray(img)
        img_transformed = self.transform(img)
        return img_transformed
    
    def get_edges(self,idx,recompute=False, threshold1=25, threshold2=50):
        if recompute:
            img_path = self.file_list[idx]
            data_dict = pickle.load(open(img_path, "rb"))
            img = data_dict['orig_frame'][:,:,::-1]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Blur the image for better edge detection
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
            # mean 255/2 * .66 for min threshold, 255/2 * 1.33 for max
            edges = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2)
            edges = cv2.GaussianBlur(edges, (13, 13), 0)
            img = Image.fromarray(edges)
            img_transformed = test_transforms(img)
        else:
            img_path = self.file_list[idx]
            data_dict = pickle.load(open(img_path, "rb"))
            img = data_dict['edges']
            img = Image.fromarray(img)
            img_transformed = self.transform(img)
        return img_transformed
    
    def get_data_dict(self,idx):
        img_path = self.file_list[idx]
        data_dict = pickle.load(open(img_path, "rb"))
        return data_dict

    def get_smoothed_fixations(self,idx):
        img_path = self.file_list[idx]
        data_dict = pickle.load(open(img_path, "rb"))
        img = data_dict['eye_smoothed']
        img = Image.fromarray(img)
        img_transformed = self.transform(img)
        return img_transformed

    def patchify_weigh_flatten(self, fixation_map, patch_size=16, num_patches=14):
        '''Takes the fixation map as input and splits it in 196 patches of size (16, 16) each,
            then calculates the sum of all pixels in each patch and create a tensor of size 196,
            where every element is the sum of the corresponding patch
        '''
        # Reshape the image tensor into patches
        patches = fixation_map.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        # Reshape the patches tensor to [num_patches^2, patch_size, patch_size]
        patches = patches.contiguous().view(num_patches * num_patches, patch_size, patch_size)
        # Calculate the sum of pixel values for each patch
        sums = patches.view(patches.size(0), -1).sum(dim=1)
        return sums

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        data_dict = pickle.load(open(img_path, "rb"))
        img = data_dict['orig_frame'][:,:,::-1] # flip color channels 
        img = Image.fromarray(img)
        img_transformed = self.transform(img)

        if self.mask_cond == 'occlude_fixation':
            eye_smoothed = self.get_smoothed_fixations(idx) # get eye data cropped etc. in model dimensions
            eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
            inverse_eye_smoothed = eye_smoothed.copy()
            viewed_mask = inverse_eye_smoothed > 0
            not_viewed_mask = inverse_eye_smoothed == 0
            not_viewed_mask = torch.tensor(np.broadcast_to(not_viewed_mask.astype(int), img_transformed.shape))
            img_transformed = torch.mul(img_transformed,not_viewed_mask)
        elif self.mask_cond == 'occlude_fixation_random':
            eye_smoothed = self.get_smoothed_fixations(idx) # get eye data cropped etc. in model dimensions
            eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
            inverse_eye_smoothed = eye_smoothed.copy()
            viewed_mask = inverse_eye_smoothed > 0
            viewed_mask, random_ang,random_x_translate,random_y_translate = apply_random_mask(viewed_mask)
            not_viewed_mask = viewed_mask != True
            not_viewed_mask = torch.tensor(np.broadcast_to(not_viewed_mask.astype(int), img_transformed.shape))
            img_transformed = torch.mul(img_transformed,not_viewed_mask)
        elif self.mask_cond == 'occlude_peripheral_fixation':
            eye_smoothed = self.get_smoothed_fixations(idx) # get eye data cropped etc. in model dimensions
            eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
            inverse_eye_smoothed = eye_smoothed.copy()
            kernel_width = 30
            kernel_height = 30
            kernel = np.ones((kernel_height,kernel_width), np.uint8) # note these get transposed below
            mask_dilated = cv2.dilate(np.moveaxis(inverse_eye_smoothed, 0, -1), kernel, iterations=1)
            viewed_mask = mask_dilated > 0
            not_viewed_mask = mask_dilated == 0
            not_viewed_mask = torch.tensor(np.broadcast_to(not_viewed_mask.astype(int), img_transformed.shape))
            img_transformed = torch.mul(img_transformed,not_viewed_mask)
        elif self.mask_cond == 'occlude_peripheral_fixation_random':
            eye_smoothed = self.get_smoothed_fixations(idx) # get eye data cropped etc. in model dimensions
            eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
            inverse_eye_smoothed = eye_smoothed.copy()
            kernel_width = 30
            kernel_height = 30
            kernel = np.ones((kernel_height,kernel_width), np.uint8) # note these get transposed below
            mask_dilated = cv2.dilate(np.moveaxis(inverse_eye_smoothed, 0, -1), kernel, iterations=1)
            viewed_mask = mask_dilated > 0
            viewed_mask, random_ang,random_x_translate,random_y_translate = apply_random_mask(viewed_mask)
            not_viewed_mask = viewed_mask != True
            not_viewed_mask = torch.tensor(np.broadcast_to(not_viewed_mask.astype(int), img_transformed.shape))
            img_transformed = torch.mul(img_transformed,not_viewed_mask)
        elif self.mask_cond == 'peripheral':
            img_transformed = data_dict['peripheral']
        elif self.mask_cond == 'random_peripheral':
            img_transformed = data_dict['peripheral']
        
        if self.dataset_type == 'vr':
            steer_path = eval(data_dict['save_trial_info']['post_steer_event_raw'])
            steer_amount = steer_path[-1] - steer_path[0]
            label = self.direction_encoded_map(steer_amount)
        else:
            if self.mask_cond == 'peripheral' or self.mask_cond == 'random_peripheral':
                label = data_dict['frame_info']['steering_dir_encoded']
            else:
                label = data_dict['save_trial_info']['steering_dir_encoded']

        fixation = self.get_smoothed_fixations(idx) # get fixation data for each sample
        fixation = torch.where(torch.isnan(fixation), torch.tensor(0.0), fixation) # replace NaN with zero       
        fixation = self.patchify_weigh_flatten(fixation)
        fixation = (fixation - torch.min(fixation)) / (torch.max(fixation) - torch.min(fixation)) #normalize weights
        input = img_transformed

        if self.model_type == 'jsf':
            fixation = self.get_smoothed_fixations(idx) # get fixation data for each sample
            fixation = torch.where(torch.isnan(fixation), torch.tensor(0.0), fixation)
            fixation = (fixation - torch.min(fixation)) / (torch.max(fixation) - torch.min(fixation))   
            fixation3d = fixation.repeat(3, 1, 1)
            # Concatenate 3d fixation map with original frame
            video = torch.cat((fixation3d.unsqueeze(0), img_transformed.unsqueeze(0)), dim=0)
            # Reshape from (frames, channels, height, width) to (channels, frames, height, width)
            input = video.permute(1, 0, 2, 3)

        if self.model_type == 'vit_fixation_only':
            fixation = self.get_smoothed_fixations(idx) # get fixation data for each sample
            fixation = torch.where(torch.isnan(fixation), torch.tensor(0.0), fixation) # replace NaN with zero
            fixation = fixation.repeat(3, 1, 1)

        return input, fixation, label


# Define helper functions
def apply_random_mask(input_viewed_mask):
    """TBD

    Inputs:
        - input_viewed_mask (string):

    Returns:
        - ???
    """
    input_viewed_mask = np.squeeze(input_viewed_mask)
    x_bounds, y_bounds = get_mask_coords(input_viewed_mask)

    centerx = input_viewed_mask.shape[0] // 2
    centery = input_viewed_mask.shape[1] // 2


    img = Image.fromarray(input_viewed_mask)
    random_ang = random.uniform(1, 100)
    im_rotate = img.rotate(random_ang)
    x_bounds, y_bounds = get_mask_coords(np.array(im_rotate))

    # define limits so that translate=(x_limit_end,y_limit_end) preserves the segmentation at the top left of the image
    # and translate=(x_limit_start,y_limit_start) at the bottom right
    x_limit_start = centerx - (x_bounds[1] - centerx)
    x_limit_end = -x_bounds[0]
    y_limit_start = centery - (y_bounds[1] - centery)
    y_limit_end = -y_bounds[0]
    random_x_translate = random.uniform(x_limit_start, x_limit_end)
    random_y_translate = random.uniform(y_limit_start, y_limit_end)
    im_rotate = img.rotate(random_ang, translate=(random_x_translate, random_y_translate))
    
    return np.array(im_rotate), random_ang, random_x_translate, random_y_translate


def peripheral_data_extraction(dataset_folder, train_data_cond):
    all_files = []
    for path in Path(f"{dataset_folder}").rglob('*.p'):
        all_files.append(str(path))

    if train_data_cond == 'peripheral':
        for img_path in all_files:
            data_dict = pickle.load(open(img_path, "rb"))
            img = data_dict['orig_frame'][:,:,::-1] # flip color channels 
            img = Image.fromarray(img)

            img_transformed = train_transforms(img)

            img = data_dict['eye_smoothed']
            img = Image.fromarray(img)
            eye_smoothed = train_transforms(img)
            eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
            inverse_eye_smoothed = eye_smoothed.copy()
            kernel_width = 30
            kernel_height = 30
            kernel = np.ones((kernel_height,kernel_width), np.uint8) # note these get transposed below
            mask_dilated = cv2.dilate(np.moveaxis(inverse_eye_smoothed, 0, -1), kernel, iterations=1)
            viewed_mask = mask_dilated > 0
            viewed_mask = torch.tensor(np.broadcast_to(viewed_mask.astype(int), img_transformed.shape))
            img_transformed = torch.mul(img_transformed,viewed_mask)
            data_dict['peripheral'] = img_transformed
            file_path = img_path.split('/',1)[1]
            Path(f"{dataset_folder}_{train_data_cond}/{file_path.rsplit('/',1)[0]}").mkdir(parents=True, exist_ok=True)
            with open(f"{dataset_folder}_{train_data_cond}/{file_path}", 'wb') as handle:
                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif train_data_cond == 'random_peripheral':
        for img_path in all_files:
            data_dict = pickle.load(open(img_path, "rb"))
            img = data_dict['orig_frame'][:,:,::-1] # flip color channels 
            img = Image.fromarray(img)

            img_transformed = train_transforms(img)

            img = data_dict['eye_smoothed']
            img = Image.fromarray(img)
            eye_smoothed = train_transforms(img)
            eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
            inverse_eye_smoothed = eye_smoothed.copy()
            kernel_width = 30
            kernel_height = 30
            kernel = np.ones((kernel_height,kernel_width), np.uint8) # note these get transposed below
            mask_dilated = cv2.dilate(np.moveaxis(inverse_eye_smoothed, 0, -1), kernel, iterations=1)
            viewed_mask = mask_dilated > 0
            viewed_mask, random_ang, random_x_translate, random_y_translate = apply_random_mask(viewed_mask)
            viewed_mask = torch.tensor(np.broadcast_to(viewed_mask.astype(int), img_transformed.shape))
            img_transformed = torch.mul(img_transformed,viewed_mask)
            data_dict['peripheral'] = img_transformed
            file_path = img_path.split('/',1)[1]
            Path(f"{dataset_folder}_{train_data_cond}/{file_path.rsplit('/',1)[0]}").mkdir(parents=True, exist_ok=True)
            with open(f"{dataset_folder}_{train_data_cond}/{file_path}", 'wb') as handle:
                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_split_data(dataset_type, data_path, train_data_cond='full', dataset_folder='vr',
                    validation=False, random_state=None):
    all_files = []

    if train_data_cond == 'full':
        for path in Path(f"{data_path}/{dataset_folder}").rglob('*.p'):
            all_files.append(str(path))
    elif train_data_cond == 'peripheral' or train_data_cond == 'random_peripheral':
        print(f'using {train_data_cond}')
        for path in Path(f"{data_path}/{dataset_folder}_{train_data_cond}").rglob('*.p'):
            all_files.append(str(path))


    if dataset_type == 'dreyeve':
        all_files_fixations = []
        for img_path in all_files:
            data_dict = pickle.load(open(img_path, "rb"))
            if not np.any(data_dict['eye_smoothed']):
                print('NO FIXATION DATA FOUND FOR ', img_path)
                continue
            all_files_fixations.append(img_path)
        all_files = all_files_fixations
        

    train_list, test_list = train_test_split(all_files, 
                                            test_size=0.2,
                                            random_state=random_state)
    if validation:
        train_list, valid_list = train_test_split(train_list, 
                                                test_size=0.15,
                                                random_state=random_state)
    else:
        valid_list = []

    print(f"Train Data: {len(train_list)}")
    print(f"Valid Data: {len(valid_list)}")
    print(f"Test Data: {len(test_list)}")
    
    return train_list, valid_list, test_list


def get_mask_coords(in_mask):
    bool_array = in_mask * 1
    lower_y = np.where(np.any(bool_array>0, axis=1))[0][0]
    higher_y = np.where(np.any(bool_array>0, axis=1))[0][-1]
    lower_x = np.where(np.any(bool_array>0, axis=0))[0][0]
    higher_x = np.where(np.any(bool_array>0, axis=0))[0][-1]
    
    return (lower_x, higher_x), (lower_y, higher_y)


def get_loaders(dataset_type, model_type, train_list, valid_list=None, test_list=None, batch_size = None, 
                train_data_cond = None, test_data_cond = None):
    # if valid_list is None: valid_list = train_data
    # if test_list is None: test_list = train_data
    train_data = MotorDataset(train_list, transform=train_transforms, mask_cond=train_data_cond, dataset_type=dataset_type, model_type=model_type)
    if len(valid_list) > 0:
        valid_data = MotorDataset(valid_list, transform=val_transforms, mask_cond=test_data_cond, dataset_type=dataset_type, model_type=model_type)
    else:
        valid_data = None
    test_data = MotorDataset(test_list, transform=test_transforms, mask_cond=test_data_cond, dataset_type=dataset_type, model_type=model_type)

    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, num_workers=16)
    if len(valid_list) > 0:
        valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True, num_workers=16)
    else: 
        valid_loader = None
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True, num_workers=16)
    
    return train_loader, valid_loader, test_loader, train_data, valid_data, test_data


