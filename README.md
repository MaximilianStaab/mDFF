# Mobile Depth From Focus (mDFF) Dataset and Toolbox

We share the mDFF dataset and its toolbox in this repository. Dataset is preprocessed as follows:

- we align and scale/crop the images using the [alignment script](python/align_focal_stacks.py).
- we project the pointclouds to the 2D image plane given the camera parameters using the [projection script](python/project_pointclouds.py).
- we generate the train/val and test datasets in .h5 file using the [dataset generation script](python/pack_to_h5.py).

## Dataset
Cropped (224x224) and clipped (0.1 to 4meters) hdf5 dataset can be downloaded [here](https://vision.in.tum.de/webarchive/hazirbas/mDFF/mDFF-dataset_cropped_clipped.h5). It contains the train, val and test data.

Original focalstacks (/focalstacks), preprocessed focalstacks (/test, /train), registered depth maps and raw pointclouds can be downloaded [here](https://vision.in.tum.de/webarchive/hazirbas/mDFF/mDFFDataset.tar.gz). 


### License
All data in the mDFF dataset is licensed under a [Creative Commons 4.0 Attribution License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Citation
Caner Hazirbas, Sebastian Georg Soyer, Maximilian Christian Staab, Laura Leal-Taixé and Daniel Cremers, _"Deep Depth From Focus"_, ACCV, 2018. ([arXiv](https://arxiv.org/abs/1704.01085))

    @InProceedings{hazirbas18ddff,
     author    = {C. Hazirbas and S. G. Soyer and M. C. Staab and L. Leal-Taixé and D. Cremers},
     title     = {Deep Depth From Focus},
     booktitle = {Asian Conference on Computer Vision (ACCV)},
     year      = {2018},
     month     = {December},
     eprint    = {1704.01085},
     url       = {https://hazirbas.com/projects/ddff/},
    }

## License
The toolbox is released under [GNU General Public License Version 3 (GPLv3)](http://www.gnu.org/licenses/gpl.html).
