import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


class CXRDataset(Dataset):

    def __init__(
            self,
            path_to_images,
            fold,
            transform=None,
            transform_bb=None,
            finding="any",
            fine_tune=False):

        self.transform = transform
        self.transform_bb = transform_bb
        self.path_to_images = path_to_images
        if not fine_tune:
            self.df = pd.read_csv("labels/nih_original_split.csv")
        else:
            self.df = pd.read_csv("labels/brixia_split.csv")
        self.fold = fold
        self.fine_tune = fine_tune

        if not fold == 'BBox':
            self.df = self.df[self.df['fold'] == fold]
        else:
            bbox_images_df = pd.read_csv("labels/BBox_List_2017.csv")
            self.df = pd.merge(left=self.df, right=bbox_images_df, how="inner", on="Image Index")

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            self.df = self.df[self.df['Finding Label'] == finding]

        self.df = self.df.set_index("Image Index")
        self.PRED_LABEL = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        if not self.fine_tune:
            label = np.zeros(len(self.PRED_LABEL), dtype=int)
            for i in range(0, len(self.PRED_LABEL)):
                # can leave zero if zero, else make one
                if self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0:
                    label[i] = self.df[self.PRED_LABEL[i].strip()
                                       ].iloc[idx].astype('int')
        else:
            ground_truth = np.array(self.df['BrixiaScoreGlobal'].iloc[idx].astype('float32'))

        if self.transform:
            image = self.transform(image)

        if self.fold == "BBox":
            # exctract bounding box coordinates from dataframe, they exist in the the columns specified below
            bounding_box = self.df.iloc[idx, -7:-3].to_numpy()

            if self.transform_bb:
                transformed_bounding_box = self.transform_bb(bounding_box)

            return image, label, self.df.index[idx], transformed_bounding_box
        elif self.fine_tune:
            return image, ground_truth, self.df.index[idx]
        else:
            return image, label, self.df.index[idx]

    def pos_neg_balance_weights(self):
        pos_neg_weights = []

        for i in range(0, len(self.PRED_LABEL)):
            num_negatives = self.df[self.df[self.PRED_LABEL[i].strip()] == 0].shape[0]
            num_positives = self.df[self.df[self.PRED_LABEL[i].strip()] == 1].shape[0]

            pos_neg_weights.append(num_negatives / num_positives)

        pos_neg_weights = torch.Tensor(pos_neg_weights)
        pos_neg_weights = pos_neg_weights.cuda()
        pos_neg_weights = pos_neg_weights.type(torch.cuda.FloatTensor)
        return pos_neg_weights


class RescaleBB(object):
    """Rescale the bounding box in a sample to a given size.

    Args:
        output_image_size (int): Desired output size.
    """

    def __init__(self, output_image_size, original_image_size):
        assert isinstance(output_image_size, int)
        self.output_image_size = output_image_size
        self.original_image_size = original_image_size

    def __call__(self, sample):
        assert sample.shape == (4,)
        x, y, w, h = sample[0], sample[1], sample[2], sample[3]

        scale_factor = self.output_image_size / self.original_image_size
        new_x, new_y, new_w, new_h = x * scale_factor, y * scale_factor, w * scale_factor, h * scale_factor
        transformed_sample = np.array([new_x, new_y, new_w, new_h])

        return transformed_sample


