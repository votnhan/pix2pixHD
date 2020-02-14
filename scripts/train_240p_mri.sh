python train.py --name tumor2braintumor_240p \
--dataroot ./datasets/mri_toys \
--netG global --last_activation tanh \
--label_nc 5 --ngf 32 --n_blocks_global 9 \
--input_nc 1 --output_nc 1 --num_D 3 \
--resize_or_crop none \
--no_vgg_loss