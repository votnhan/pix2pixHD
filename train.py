import time
import os
import numpy as np
import torch
import logging
from torch.autograd import Variable
from collections import OrderedDict
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

logging.basicConfig(level=logging.INFO, filename='saved/info.log', filemode='w',
                    format='%(levelname)s - %(message)s')
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
if opt.fp16:    
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')             
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
n_samples = len(dataset)
n_iter = n_samples // opt.batchSize

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        # save_fake = total_steps % opt.display_freq == display_delta
        save_fake = i % n_iter == 0

        ############## Forward Pass ######################
        losses, generated = model(Variable(data['label']), Variable(data['inst']), 
            Variable(data['image']), Variable(data['feat']), infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        if opt.gan_type == 'wgan-gp':
            loss_D = loss_dict['d_fake'] + loss_dict['d_real'] + opt.lambda_gp*loss_dict['gp']
        else:
            loss_D = (loss_dict['d_fake'] + loss_dict['d_real']) * 0.5
        
        loss_G = opt.lambda_rectr*loss_dict['g_gan'] + (1-opt.lambda_rectr)*loss_dict['rectr']
        loss_G += loss_dict.get('g_gan_feat', 0) + loss_dict.get('g_vgg', 0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
        else:
            loss_G.backward()          
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
        else:
            loss_D.backward()        
        optimizer_D.step()        

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ### display output images
        if save_fake:
            mean = data['mean']
            lbl = ('input_label', util.tensor2label(data['label'][0], opt.label_nc))
            syn_img = ('synthesized_image', util.tensor2MRI(generated.data[0]*mean.cuda(),
                        scale=False, **opt.statistics))
            real_img = ('real_image', util.tensor2MRI(data['image'][0]*mean,
                        scale=False, **opt.statistics))
            visuals = OrderedDict([lbl, syn_img, real_img])
            visualizer.display_current_results(visuals, epoch, total_steps)
            img_file = os.path.basename(data['path'][0])
            logging.info('Save image: {} at epoch {}'.format(img_file, epoch))

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
