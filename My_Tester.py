
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from SiamUnet_diff import *
from SiamUnet_diff_With_Matching import *


from My_Trainer import test_loader

# Set CUDA_LAUNCH_BLOCKING environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import random

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


############## Test the model ##################
############## Test the model ##################

############## Confusion Matrix ##################

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


############## Test the model ##################
from torchvision.utils import save_image

if __name__ == "__main__":


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"


    print(device)

    model = SiamUnet_diff_Match(input_nbr=3, label_nbr=1)
    model=model.to(device)
    model.load_state_dict(torch.load("E://VS Projects//test_25-2-2025_FC//checkpoint9-withMatch2//ResUnet199.pth"))

    test_results_path="E://VS Projects//test_25-2-2025_FC//test_results9-withMatch"
    os.makedirs(test_results_path,exist_ok=True)
    TP=0
    TN=0
    FP=0
    FN=0
    for _, data in enumerate(test_loader):
        pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
        pre_tensor = pre_tensor.to(device)
        post_tensor = post_tensor.to(device)
        label_tensor = label_tensor.to(device)
        probs = model(pre_tensor, post_tensor)
        prediction = torch.where(probs>0.5,1.0,0.0)
        true_positives, false_positives, true_negatives, false_negatives = confusion(prediction,label_tensor)
        TP+=true_positives
        TN+=true_negatives
        FP+=false_positives
        FN+=false_negatives
        for i in range(prediction.shape[0]):
            save_image(prediction[i,:,:,:].cpu(), os.path.join(test_results_path, fname[i]))


    ################## Visualize the results ##################
    import matplotlib.pyplot as plt
    import numpy as np

    pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["label"], data["fname"]
    fig=plt.figure(figsize=(15,5))

    preplot=fig.add_subplot(341)
    preplot.imshow(pre_tensor[1,:,:,:].permute(1,2,0).numpy())
    preplot.set_title("pre-change")

    postplot=fig.add_subplot(342)
    postplot.set_title("post-change")
    postplot.imshow(post_tensor[1,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(343)
    postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[1,:,:,:].permute(1,2,0).numpy())

    labelplot=fig.add_subplot(344)
    labelplot.set_title("prediction")
    labelplot.imshow(prediction[1,:,:,:].permute(1,2,0).cpu().numpy())
    # transforms.ToPILImage()(pre_tensor[0,:,:,:])
    # transforms.ToPILImage()(post_tensor[0,:,:,:])
    # transforms.ToPILImage()(label_tensor[0,:,:,:])



    preplot=fig.add_subplot(345)
    preplot.imshow(pre_tensor[2,:,:,:].permute(1,2,0).numpy())
    preplot.set_title("pre-change")

    postplot=fig.add_subplot(346)
    postplot.set_title("post-change")
    postplot.imshow(post_tensor[2,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(347)
    postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[2,:,:,:].permute(1,2,0).numpy())

    labelplot=fig.add_subplot(348)
    labelplot.set_title("prediction")
    labelplot.imshow(prediction[2,:,:,:].permute(1,2,0).cpu().numpy())
    print(f'fname={fname[0]}')


    preplot=fig.add_subplot(349)
    preplot.imshow(pre_tensor[3,:,:,:].permute(1,2,0).numpy())
    preplot.set_title("pre-change")

    postplot=fig.add_subplot(3,4,10)
    postplot.set_title("post-change")
    postplot.imshow(post_tensor[3,:,:,:].permute(1,2,0).numpy())

    postplot=fig.add_subplot(3,4,11)
    postplot.set_title("label-tensor")
    postplot.imshow(label_tensor[3,:,:,:].permute(1,2,0).numpy())

    labelplot=fig.add_subplot(3,4,12)
    labelplot.set_title("prediction")
    labelplot.imshow(prediction[3,:,:,:].permute(1,2,0).cpu().numpy())
    print(f'fname={fname[0]}')

    plt.show()

    ############# Calculate the metrics ##############
    OA=(TP+TN)/(TP+TN+FP+FN)
    Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    F1_score=2*Precision*Recall/(Precision+Recall)
    IoU =TP/(TP+FP+FN)
    print(f'OA={OA:.3f}, Precision={Precision:.3f}, Recall={Recall:.3f}, IoU={IoU:.3f}, F1-score={F1_score:.3f}' )