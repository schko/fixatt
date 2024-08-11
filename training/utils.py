# Imports
import torch
import timm
import pickle
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

from PIL import Image

# Add TimeSformer library to the path
sys.path.append('../TimeSformer-main')
from timesformer.models.vit import TimeSformer

# from .dataloaders import apply_random_mask, train_transforms
from dataloaders import apply_random_mask, train_transforms

# Define Custom Loss Function
class CustomLoss(nn.Module):
    '''Custom loss function modifies the default CrossEntropyLoss (CELoss)
       to account for the average intersection between attention 
       maps of all heads of the ViT model and human fixation maps.

       Take the average intersection for every sample in the batch,
       pass it from a sigmoid function and multiply the CELoss for 
       each sample with (1 / avg_intersection), where avg_intersection is
       the corresponding average intersection of each sample.

       new_loss = (1 - lambda) * CELoss + lambda * (1 / sigmoid(avg_intersection))

       Inputs:
            - output:           (tensor) [batch_size, C], C:number of classes (2 in our case)
            - target:           (tensor) [batch_size], class indices (0 or 1 in our case)
            - avg_intersection: (tensor) [batch_size], a tensor containg the average intersection
                                                for each sample of the batch

        Returns:
            - new_loss:         (tensor) [batch_size], the average of the new_loss values of each sample
    '''
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, avg_intersection, l):

        avg_intersection = nn.Sigmoid()(avg_intersection) # remove sigmoid in version 2 
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(output, target)
        new_loss = (1 - l) * loss + l *  (1 /  avg_intersection)
        return torch.mean(new_loss)


# Define EarlyStopper class
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def patchify_weigh_flatten(fixation_map, patch_size=16, num_patches=14):
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


def plot_sample_image(idx, plot_cond, test_data):
    match plot_cond:
        case 'orig_frame':
            plt.figure()
            plt.imshow(test_data.__getitem__(idx)[0].permute(1,2,0))
        case 'fixation_heatmap':
            fixations = test_data.get_smoothed_fixations(idx)[0]
            plt.figure()
            plt.imshow(fixations)
        case 'fixation_reduced':
            fixation = test_data.get_smoothed_fixations(idx)
            fixation = torch.where(torch.isnan(fixation), torch.tensor(0.0), fixation) # replace NaN with zero
            fixation = patchify_weigh_flatten(fixation)
            fixation = (fixation - torch.min(fixation)) / (torch.max(fixation) - torch.min(fixation)) #normalize weights
            fixation = torch.reshape(fixation, (14, 14))
            fixation = np.array(fixation)

            plt.figure()
            # Set custom tick positions and labels for both axes
            custom_ticks = [0, 4, 8, 12]
            plt.xticks(custom_ticks)
            plt.yticks(custom_ticks)
            plt.tick_params(axis='both', labelsize=20)         
            plt.imshow(fixation)
            # plt.savefig('fixation_reduced.png', transparent=True, bbox_inches="tight", dpi=300)
        case 'fixation_overlaid':
            input_image = test_data.__getitem__(idx)[0]
            fixation = test_data.get_smoothed_fixations(idx) # get fixation data for each sample
            fixation = torch.where(torch.isnan(fixation), torch.tensor(0.0), fixation) # replace NaN with zero
            fixation = fixation[0]
            fixation_np = fixation.numpy()
            input_image_np = input_image.numpy().transpose(1, 2, 0)

            # Normalize arrays to [0, 1] for proper visualization
            fixation_np = (fixation_np - fixation_np.min()) / (fixation_np.max() - fixation_np.min())
            input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())
            # Set the opacity for the overlay (0.6 means 60% transparency)
            opacity = 0.5
            # Plot the input_image_np as the background
            plt.imshow(input_image_np)
            # Plot the 'fixation_np' on top of the input_image_np with transparency (alpha)
            # The 'cmap' argument specifies the colormap to use for 'fixation_np'
            # The 'alpha' argument controls the transparency level
            plt.imshow(fixation_np, alpha=opacity)
            # Add a colorbar for the 'fixation_np' array only
            cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(cmap='viridis'))
            cbar.set_label("Fixation Values")

            # Set custom tick positions and labels for both axes
            custom_ticks = [0, 50, 100, 150, 200]
            plt.xticks(custom_ticks)
            plt.yticks(custom_ticks)
            plt.tick_params(axis='both', labelsize=20) 

            # plt.savefig('fixation_overlaid.png', transparent=True, bbox_inches="tight", dpi=300)

            # Show the plot
            plt.figure()
            plt.show()
        case 'fixation_mask':
            img_transformed = test_data.__getitem__(idx)[0]
            eye_smoothed = test_data.get_smoothed_fixations(idx)[0]
            eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
            inverse_eye_smoothed = eye_smoothed.copy()
            viewed_mask = inverse_eye_smoothed > 0
            not_viewed_mask = inverse_eye_smoothed == 0
            not_viewed_mask = torch.tensor(np.broadcast_to(not_viewed_mask.astype(int), img_transformed.shape))
            img_transformed = torch.mul(img_transformed,not_viewed_mask)
            plt.figure()
            plt.imshow(viewed_mask)
        case 'fixation_occlusion':
            img_transformed = test_data.__getitem__(idx)[0]
            eye_smoothed = test_data.get_smoothed_fixations(idx)[0]
            eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
            inverse_eye_smoothed = eye_smoothed.copy()
            viewed_mask = inverse_eye_smoothed > 0
            not_viewed_mask = inverse_eye_smoothed == 0
            not_viewed_mask = torch.tensor(np.broadcast_to(not_viewed_mask.astype(int), img_transformed.shape))
            img_transformed = torch.mul(img_transformed,not_viewed_mask)
            plt.figure()
            plt.imshow(img_transformed.permute(1,2,0))
        case 'random_fixation_occlusion':
            img_transformed = test_data.__getitem__(idx)[0]
            eye_smoothed = test_data.get_smoothed_fixations(idx)[0]
            eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
            inverse_eye_smoothed = eye_smoothed.copy()

            viewed_mask = inverse_eye_smoothed > 0
            viewed_mask, random_ang,random_x_translate,random_y_translate = apply_random_mask(viewed_mask)
            not_viewed_mask = viewed_mask != True

            not_viewed_mask = torch.tensor(np.broadcast_to(not_viewed_mask.astype(int), img_transformed.shape))
            img_transformed = torch.mul(img_transformed,not_viewed_mask)
            plt.figure()
            plt.imshow(img_transformed.permute(1,2,0))
        case 'peripheral_mask':
            data_dict = test_data.get_data_dict(idx)
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
            plt.figure()
            plt.imshow(viewed_mask)
        case 'peripheral':
            data_dict = test_data.get_data_dict(idx)
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
            plt.figure()
            plt.imshow(img_transformed.permute(1,2,0))
        case 'random_peripheral':
            data_dict = test_data.get_data_dict(idx)
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
            viewed_mask, random_ang,random_x_translate,random_y_translate = apply_random_mask(viewed_mask)
            viewed_mask = torch.tensor(np.broadcast_to(viewed_mask.astype(int), img_transformed.shape))
            img_transformed = torch.mul(img_transformed,viewed_mask)
            plt.figure()
            plt.imshow(img_transformed.permute(1,2,0))


def get_new_model_configs(vit_version, model_type, subset_layers=False, device='cpu'):
    if model_type == 'jsf':
        # TimeSformer pretrained weights: 'fixatt/TimeSformer-main/timesformer/pretrained_models/TimeSformer_divST_8x32_224_K600.pyth'
        # pretrained_model_path = 'fixatt/TimeSformer-main/timesformer/pretrained_models/TimeSformer_divST_8x32_224_K400.pyth'
        pretrained_model_path = '' # in this case, it used pretrained weights from ViT model
        model = TimeSformer(img_size=224, num_classes=2, num_frames=2, attention_type='joint_space_time',  pretrained_model=pretrained_model_path).to(device)
    else:
        model = timm.create_model(vit_version, pretrained=True, num_classes=2).to(device)
    
    if subset_layers:
        model.blocks = nn.Sequential(*[model.blocks[i] for i in range(subset_layers)])

    if model_type == 'fax':
        criterion = CustomLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    return model, criterion, optimizer


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def train_one_epoch(model_type, model, criterion, optimizer, loader, input_data, device, batch_size, l):
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(loader, 0):
        inputs, fixation, labels = data # get the inputs; data is a list of [inputs, fixation, labels]
        inputs = inputs.to(device)
        fixation = fixation.to(device)
        labels = labels.to(device)
        optimizer.zero_grad() # zero the parameter gradients

        if model_type == 'fax':
            # forward + backward + optimize
            outputs, intersections = model(inputs, fixation)        
            avg_intersection = torch.mean(intersections, dim=1).to(device)
            loss = criterion(outputs, labels, avg_intersection, l)
            loss.backward()
            optimizer.step()
        elif model_type == 'vit_fixation_only':
            # forward + backward + optimize
            outputs = model(fixation) # use only fixation map as input to the model
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # evaluate
        total_loss += loss.item()  * batch_size
        _, y_pred_tags = torch.max(outputs, dim = 1) 
        correct += (y_pred_tags == labels).float().sum()
    train_accuracy = 100 * (correct.item() / len(input_data))
    
    return total_loss, train_accuracy


def eval_dataset(model_type, model, criterion, loader, input_data, device, batch_size, l, export_preds = False):
    correct = 0
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []
    model = model.to(device)
    with torch.no_grad():
        if device != 'cpu':
            net =  (model) if batch_size > 10 else model
        else:
            net = model
        for i, data in enumerate(loader):
            inputs, fixation, labels = data # get the inputs; data is a list of [inputs, fixation, labels]
            all_labels.extend(list(labels.numpy()))
            inputs = inputs.to(device)
            fixation = fixation.to(device)
            labels = labels.to(device) 

            if model_type == 'fax':           
                outputs, intersections = net(inputs, fixation)
                avg_intersection = torch.mean(intersections, dim=1).to(device)
                total_loss += criterion(outputs, labels, avg_intersection, l).item() * batch_size
            elif model_type == 'vit_fixation_only':
                outputs = net(fixation) # use only fixation map as input to the model
                total_loss += criterion(outputs, labels).item() * batch_size
            else:
                outputs = net(inputs)
                total_loss += criterion(outputs, labels).item() * batch_size
            
            if export_preds:
                all_probs.extend(outputs.data.cpu().detach().numpy())
            _, predicted = torch.max(outputs.data, 1)
            if export_preds:
                all_preds.extend(predicted.cpu().detach().numpy())
            correct += (predicted == labels).sum()
        accuracy = 100 * (correct.item()) / len(input_data)
    
    return np.array(all_probs), np.array(all_preds), accuracy, total_loss


def compute_image_contrast(data_dict):

    img = data_dict['orig_frame']
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # separate channels
    L, A, B = cv2.split(lab)

    # compute minimum and maximum in 5x5 region using erode and dilate
    kernel = np.ones((5, 5), np.uint8)
    min = cv2.erode(L, kernel, iterations = 1)
    max = cv2.dilate(L, kernel, iterations = 1)

    # convert min and max to floats
    min = min.astype(np.float64) 
    max = max.astype(np.float64) 

    # compute local contrast
    contrast = (max - min) / (max + min)

    # get average across whole image
    average_contrast = 100 * np.mean(contrast)

    return average_contrast


def extract_trial_info(dataset_type, data_idx, train_data, valid_data, test_data, data_type='test'):
    '''
    Args:
        dataset_type (str): type of dataset, 'vr' or 'dreyeve'
        data_idx (int): Index of file in spllit train/valid/test data (list)
        data_type (str): Type of data. It will be either 'train', 'valid', 'test'.
    return:
        trial_info_dict (dict): Dictionary includes the trial density, participant's session and trial informaiton, 
                                class of the trial (Left or right), fixation map relates to the trial.
    '''

    # determine the type of data we are extracting the trial information for and select data accordingly 
    if data_type == 'test':
        extract_data = test_data
    elif data_type == 'train':
        extract_data = train_data
    elif data_type == 'valid': 
        extract_data = valid_data
    else:
        raise Exception("Invalid data type either. Should be either 'train', 'valid', or 'test'.")

    # load the dictionary from .pkl file for the trial from the selected dataset based on the index
    rel_image = extract_data.file_list[data_idx]
    data_dict = pickle.load(open(rel_image, "rb"))
    x, fixation, label = extract_data.__getitem__(data_idx)

    # extract the relevant information from the trial information dictionary
    trial_info_dict = {}

    if dataset_type == 'vr':
        # extract participant session and trial information
        ppid_no = data_dict['save_trial_info']['ppid']
        session_no = data_dict['save_trial_info']['session']
        trial_no = data_dict['save_trial_info']['trial']
        trial_info_dict['participant_trial_info'] = f'{ppid_no}_{session_no}_{trial_no}'
        # extract trial density
        trial_info_dict['trial_density'] = data_dict['save_trial_info']['density']
        trial_info_dict['label'] = label
        # cropped raw image and fixation overlay
        cropped_im_array = x.detach().cpu().numpy()
        cropped_im_array = cropped_im_array.transpose(1, 2, 0)
        eye_smoothed = extract_data.get_smoothed_fixations(data_idx) # get eye data cropped etc. in model dimensions
        eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
        trial_info_dict['fixation_map'] = eye_smoothed # extract fixation map
    
    elif dataset_type == 'dreyeve':
        trial_info_dict['trial_contrast'] = compute_image_contrast(data_dict)
        ppid_no = data_dict['save_trial_info']['participant']
        trial_no = data_dict['save_trial_info']['trial']
        premotor_frame_no = data_dict['save_trial_info']['premotor_frame']
        trial_info_dict['participant_trial_info'] = f'{ppid_no}_{trial_no}_{premotor_frame_no}'
        trial_info_dict['label'] = label
        # cropped raw image and fixation overlay
        cropped_im_array = x.detach().cpu().numpy()
        cropped_im_array = cropped_im_array.transpose(1, 2, 0)
        eye_smoothed = extract_data.get_smoothed_fixations(data_idx) # get eye data cropped etc. in model dimensions
        eye_smoothed = np.nan_to_num(np.array(eye_smoothed))
        trial_info_dict['fixation_map'] = eye_smoothed # extract fixation map
        # Edge
        edges = extract_data.get_edges(data_idx, recompute=True) # this describes presence of an edge at pixel
        edges = np.nan_to_num(np.array(edges))
        trial_info_dict['edges'] = edges

    return trial_info_dict
