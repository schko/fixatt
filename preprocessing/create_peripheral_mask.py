import cv2
import numpy as np
import torch

def create_peripheral_mask(input_frame, fixations_smoothed):
    """
    Create a peripheral mask based on eye fixations to highlight viewed regions in an image.

    Args:
        input_frame (numpy.ndarray): The image to be masked.
        fixations_smoothed (numpy.ndarray): Eye fixation heatmap.

    Returns:
        torch.Tensor: Masked image with peripheral regions highlighted.
    """

    # Define kernel dimensions for dilation
    kernel_width = 30
    kernel_height = 30

    # Create a square kernel for dilation
    dilation_kernel = np.ones((kernel_height, kernel_width), np.uint8)

    # Dilate the eye fixation heatmap
    dilated_heatmap = cv2.dilate(np.moveaxis(fixations_smoothed, 0, -1), dilation_kernel, iterations=1)

    # Create a mask indicating viewed regions
    viewed_mask = dilated_heatmap > 0

    # Broadcast and convert the mask to a tensor to match the image shape
    broadcasted_viewed_mask = torch.tensor(np.broadcast_to(viewed_mask.astype(int), input_frame.shape))

    # Apply the viewed mask to the input image
    masked_transformed_img = torch.mul(input_frame, broadcasted_viewed_mask)
    
    return masked_transformed_img