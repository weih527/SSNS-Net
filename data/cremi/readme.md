# CREMI

It contains the following 10 h5 files:

cremiA_inputs_interp.h5

cremiA_labels.h5

cremiA+_inputs_interp.h5

cremiB_inputs_interp.h5

cremiB_labels.h5

cremiB+_inputs_interp.h5

cremiC_inputs_interp.h5

cremiC_labels.h5

cremiC+_inputs_interp.h5

*masks_cremi_suhu.h5: The cutout masks used for validation



Note that:

1) The size of each volume is 1250x1024x1024. They are obtained by the way of center cropping from [CREMI raw volumes](https://cremi.org/).
2) The '_interp' means that the noisy images in raw volumes are restored by the way of frame interpolation ([Learning to Restore ssTEM Images from Deformation and Corruption](https://link.springer.com/chapter/10.1007/978-3-030-66415-2_26)).