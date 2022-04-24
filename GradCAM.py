import os

from tensorflow import keras
import keras.backend as K 

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import databases

# INPUT VARIABLES ========================================================#

database_file_name = "RegistrationDatabase_3_frames_2CH_4CH_CAMUS_Demonstrator.hdf5"

# list of convolutional layers in respect to which we want to calculate gradients
# commonly last convolution layer
conv_list = ['activation_17', 'activation_17', 'activation_17'] 

# list of outputs 
# in this particular case we had last convolution layer 'activation_17' followed 
# by deep layers and 3 outputs
# so we want to explore activation of last layer related to all 3 outputs
# make sure to create pairs of elements from conv_list and ch_list, i.e that shapes of the two lists match
ch_list = [0, 1, 2]

# folder to store figures
folder_store = 'GradCAMS_1406_activation_layer'
# folder where the trained model is stored
folder_model = '1406_filters16'

# create folder to store figures if it does not exist
if not os.path.exists(folder_store):
    os.makedirs(folder_store)

# Path and filename to load data 
respiratory_base_path = os.path.join(
    os.path.abspath(os.path.join(__file__, '../..')),
    'Databases'
)
db_path = os.path.join(
        respiratory_base_path,
        'Sequence_2021-03-05_10-54-23',
        'V202103051_data.sqlite'
    )


# Path to trained model
storage_folder = os.path.join(
    os.path.abspath(os.path.join(__file__, '../..')),
    'Results', folder_model
)

# define how many predictions/images we want to test
n_iter=1000 
# the following are application-related variables, not related to GC
bins=20 
first_image_id=41
last_image_id=441 
angle_max=0 
translation_max=80 

# ==========================================================================#

# read the data
db = databases.DemonstratorDatabase(db_path, bins) # application-related
binned_images = db.bin_images(first_image_id, last_image_id) # binning is application-related

# read the trained model
model = keras.models.load_model(os.path.join(storage_folder, 'model.h5'), compile=False)
print(model.summary())

#--------------------------------- GradCAMS ---------------------------------#

for i in range(n_iter): 
    # we are doing GC for n_iter number of different inputs
    # follwing lines of code are aimd to generate inputs to CNN
    # in your case it would be any input your network is trained to work on
    alpha = np.random.uniform(-angle_max, angle_max)
    beta = np.random.uniform(-angle_max, angle_max)
    translation_x = np.random.uniform(-translation_max, translation_max)
    translation_y = np.random.uniform(-translation_max, translation_max)
    translation_z = np.random.uniform(-translation_max, translation_max)
    
    bin = i % bins
    fixed_image_id, moving_image_id = np.random.choice(
        a=binned_images[bin][0],
        size=2
    )
    fixed, moving, alpha_gt, translation_x_gt, translation_y1_gt = db.get_sub_sectors(
        fixed_image_id, moving_image_id, alpha, translation_x, translation_y
    )
    
    # this particular model takes 2 images as input
    # before calculating GradCams we need to make predictions on a particular input 
    model.predict([fixed.reshape((1, 512, 768, 1)), moving.reshape((1, 512, 768, 1))])

    heatmaps = []
    for ch, conv_name in zip(ch_list, conv_list):
        
        # take desired convolutional layer defined above
        last_conv = model.get_layer(conv_name)

        #----------- STEP1: compute gradients

        # calculate the gradient of a specific output 
        # with respect to the feature maps of the specific convolutional layer  
        grads = K.gradients(model.output[ch], last_conv.output)[0]

        #----------- STEP2: average gradients

        # pool the gradients over the width and height
        pooled_grads = K.mean(grads, axis=(0,1,2))
        iterate = K.function([model.input[0],model.input[1]],[pooled_grads,last_conv.output[0]])
        pooled_grads_value, conv_layer_output = iterate([fixed.reshape((1, 512, 768, 1)),moving.reshape((1, 512, 768, 1))])

        #----------- STEP3: perform a weighted sum

        # multiply feature maps by corresponding weights
        for i in range(np.size(pooled_grads_value)):
            conv_layer_output[:,:,i] *= pooled_grads_value[i]
        
        # take mean value to obtain heatmap
        heatmap = np.mean(conv_layer_output,axis=-1)
    
        # take absolute value because negative gradient matters as much as positive one
        heatmap = abs(heatmap)

        heatmaps.append(heatmap) 

    # scale heatmaps based on the global max value
    heatmaps[0] /= np.max(heatmaps)
    heatmaps[1] /= np.max(heatmaps)
    heatmaps[2] /= np.max(heatmaps)

    #----------- VISUALIZE:
    # now we have our GC ready to be visualized
    # 
    # the best way to visualize is to upsample GradCam images to 
    # size of input images and plot them on top of inputs
    # so that we really see where the decision comes from
     
    upsample_R = cv.resize(heatmaps[0], (768,512))
    upsample_TX = cv.resize(heatmaps[1], (768,512))
    upsample_TY = cv.resize(heatmaps[2], (768,512))

    fig1, axs = plt.subplots(2, 4, figsize=(20,8))
    axs[0,0].title.set_text("Fixed")
    axs[0,0].imshow(fixed/255, cmap="gray")

    axs[0,1].title.set_text("Fixed R")
    axs[0,1].imshow(fixed, cmap="gray")
    axs[0,1].imshow(upsample_R,alpha=0.5)

    axs[0,2].title.set_text("Fixed TX")
    axs[0,2].imshow(fixed, cmap="gray")
    axs[0,2].imshow(upsample_TX,alpha=0.5)

    axs[0,3].title.set_text("Fixed TY")
    axs[0,3].imshow(fixed, cmap="gray")
    axs[0,3].imshow(upsample_TY,alpha=0.5)

    axs[1,0].title.set_text("Moving")
    axs[1,0].imshow(moving, cmap="gray")

    axs[1,1].title.set_text("Moving R")
    axs[1,1].imshow(moving, cmap="gray")
    axs[1,1].imshow(upsample_R,alpha=0.5)

    axs[1,2].title.set_text("Moving TX")
    axs[1,2].imshow(moving, cmap="gray")
    axs[1,2].imshow(upsample_TX,alpha=0.5)

    axs[1,3].title.set_text("Moving TY")
    axs[1,3].imshow(moving, cmap="gray")
    axs[1,3].imshow(upsample_TY,alpha=0.5)

    fig1.savefig(folder_store + '/sample' + str(i) + '.png')
    plt.close()
