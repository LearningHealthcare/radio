""" Batch class CTImagesSegmentationBatch for storing CT-scans with segmentation masks. """

import numpy as np
import logging
import os

try:
    import pydicom as dicom # pydicom library was renamed in v1.0
except ImportError:
    import dicom

from .ct_masked_batch import CTImagesMaskedBatch
from ..dataset import action, DatasetIndex, SkipBatchException  # pylint: disable=no-name-in-module

# logger initialization
logger = logging.getLogger(__name__) # pylint: disable=invalid-name

AIR_HU = -2000
DARK_HU = -2000
search_for_no_dvt = ['ePAD DSO-NO DVT',
                     'ePAD DSO-NODVT',
                     'ePAD DSO-no DVT',
                     'ePAD DSO- NO DVT',
                     'ePAD DSO-NO DNT',
                     'ePAD DSO-NO  DVT',
                     'ePAD DSO-NI DVT',
                     'ePAD DSO-noDVT',
                     'ePAD DSO-NO DVT',
                     'ePAD DSO-NO DVT']

class CTImagesSegmentationBatch(CTImagesMaskedBatch):
    """ Batch class for storing batch of ct-scans with masks based on segmentations

        Allows to load masks from Dicom segmentation files (ePAD). Masks are stored in self.masks

        Parameters
        ----------
        index : dataset.index
            ids of scans to be put in a batch

        Attributes
        ----------
        components : tuple of strings.
            List names of data components of a batch, which are `images`,
            `masks`, `origin` and `spacing`.
            NOTE: Implementation of this attribute is required by Base class.
        images : ndarray
            contains ct-scans for all patients in batch.
        masks : ndarray
            contains masks for all patients in batch.
        filenames : ndarray
            contains list of filenames for dicom images (ct-scans)
        segmentation_path : basestring
            defines that path where we will look for segmentations
        segmentation_map : dictionary
            contains a map of image filename to segmentation filename
        """

    components = "images", "masks", "spacing", "origin", "filenames", "segmentation_path", "segmentation_map"

    def __init__(self, index, *args, **kwargs):
        """ Execute Batch construction and init of basic attributes

        Parameters
        ----------
        index : Dataset.Index class.
            Required indexing of objects (files).
        """
        super().__init__(index, *args, **kwargs)
        self.masks = None

    def extract_mask(self, reference_image_filename, i):
        # get the dictionary for this CT slice if the it's in the map. if not, return immediately
        if reference_image_filename in self.segmentation_map.keys():
            d = self.segmentation_map[reference_image_filename]
        else:
            return

        # loop through the list of segmentations for this CT slice
        for j in range(0, len(d['segmentation_file'])):
            segmentation = d['segmentation_file'][j]
            # if the segmentation is blank, want to skip it
            if segmentation == '':
                continue
            else:
                # check if the label says no DVT is present. If not, continue, we don't want to
                if any(s in d['label'][j] for s in search_for_no_dvt):
                    continue
                # check if the label says DVT is present. If so,
                elif 'ePAD DSO-DVT' in d['label'][j]:
                    slice_number = d['slice_number'][j]
                    print('reading segmentation file: ' + segmentation)
                    segmentation_dcm = dicom.dcmread(os.path.join(self.segmentation_path, segmentation))
                    segmentation_mask = segmentation_dcm.pixel_array[slice_number]
                    # check to make sure segmentation exists
                    if segmentation_mask is None or segmentation_mask == '':
                        print('Segmentation file found with no segmentation mask. No mask saved')
                        continue
                    # finally add legitimate segmentation masks
                    if self.masks[i].shape == segmentation_mask.shape:
                        self.masks[i] = self.masks[i] + segmentation_mask
                    else:
                        print("ERROR: Segmentation mask size is off. "+str(segmentation_mask.shape))
                elif 'ePAD DSO-Lesion' in d['label'][j]:
                    # if we hit this case, continue. I'm not sure what this means for now
                    continue
                else:
                    print("ERROR: weird label"+d['label'][j])

    @action
    def create_mask(self):
        """ Create `masks` component from dictionary

        Notes
        -----
        dictionary d should allow be structured to allow us to say
        d['filename of current slice'] to get the filename and slice of
        the segmentation dicom
        """
        if self.segmentation_map is None:
            logger.warning("segmentation_map must exist" +
                           "and be set before calling this method. " +
                           "Nothing happened.")

        # initialize masks to be all 0s
        self.masks = np.zeros_like(self.images)

        # use filenames to read in segmentation from annotations
        for i in range(0, len(self.images)):
            mask = self.extract_mask(self.filenames[i], i)
            #print('checking for segmentation file for slice #'+str(i))

        print('completed loading masks for current batch')
        return self

    @action
    def set_segmentation_path(self, seg_path):
        if seg_path is None:
            logger.log("seg_path should not be None. Nothing happened")
        else:
            self.segmentation_path = seg_path
        return self

    @action
    def set_segmentation_map(self, seg_map):
        if seg_map is None:
            logger.log("seg_map should not be None. Nothing happened")
        else:
            self.segmentation_map = seg_map
        return self