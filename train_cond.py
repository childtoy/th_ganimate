import os
import sys
import torch
from dataset.motion import MotionData, load_multiple_dataset
from models import create_model, create_conditional_model, get_group_list
from models.architecture import get_pyramid_lengths, joint_train, adjust_label_list
from models.utils import get_interpolator
from option import TrainOptionParser
from os.path import join as pjoin
import time
from torch.utils.tensorboard import SummaryWriter
from loss_recorder import LossRecorder
from demo import load_all_from_path
from utils import get_device_info
from data_utils.data_util import load_sin_motion, lerp_sample
import numpy as np 
import sys 
import torch.nn.functional as F

def main():
    start_time = time.time()

    parser = TrainOptionParser()
    args = parser.parse_args()
    args.device = torch.device('cuda:0')
    device = args.device
    # cpu_str, gpu_str = get_device_info()
    # print(f'CPU :{cpu_str}\nGPU: {gpu_str}')

    parser.save(pjoin(args.save_path, 'args.txt'))
    os.makedirs(args.save_path, exist_ok=True)

    motion, motion_data = load_sin_motion(args)
    # motion_data = MotionData(pjoin(args.bvh_prefix, f'{args.bvh_name}.bvh'),
    #                             padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
    #                             contact=args.contact, keep_y_pos=args.keep_y_pos,
    #                             joint_reduction=args.joint_reduction)
    motion = motion.permute(1,0,2)
    multiple_data = [motion]
    labels = np.array(motion_data['labels'])
    # print(labels)
    # motion = motion.repeat(args.batch_size,1,1,1)
    num_channels = motion.shape[1]
    labels = torch.from_numpy(labels).to(torch.float32)
    labels = labels.to(args.device).repeat(args.batch_size,1)
    neibor_list = motion_data['neighbor_list']
    interpolator = get_interpolator(args)
    lengths = []
    min_len = 10000

    adjusted_label_list = []
    adjusted_fake_label_list = []
    
    for i in range(len(multiple_data)):
        new_length = get_pyramid_lengths(args, multiple_data[i].shape[-1])
        
        min_len = min(min_len, len(new_length))
        if args.num_stages_limit != -1:
            new_length = new_length[:args.num_stages_limit]
        lengths.append(new_length)
    for lens in lengths[0]:
        adjusted_labels = adjust_label_list(np.array(motion_data['labels']), lens)
        adjusted_labels = torch.Tensor(adjusted_labels)
        adjusted_labels = F.one_hot(torch.Tensor(adjusted_labels).to(torch.int64), num_classes=3).to(args.device).unsqueeze(0).permute(0,2,1).float()
        adjusted_label_list.append(adjusted_labels)
        # adjusted_labels = adjust_label_list(np.random.randint(0,3,(50,)), lens)
        adjusted_fake_labels = torch.Tensor(np.random.randint(0,3,(lens,)))
        adjusted_fake_labels = F.one_hot(adjusted_fake_labels.to(torch.int64), num_classes=3).to(args.device).unsqueeze(0).permute(0,2,1).float()
        adjusted_fake_label_list.append(adjusted_fake_labels)


    for i in range(len(multiple_data)):
        lengths[i] = lengths[i][-min_len:]
    if not args.silent:
        print('Levels:', lengths)

    log_path = pjoin(args.save_path, './logs')
    if os.path.exists(log_path):
        os.system(f'rm -r {log_path}')
    writer = SummaryWriter(pjoin(args.save_path, './logs'))
    loss_recorder = LossRecorder(writer)

    # if args.path_to_existing and args.conditional_generator:
    #     ConGen = load_all_from_path(args.path_to_existing, args.device, use_class=True)
    # else:
    ConGen = None

    gans = []
    gens = []
    amps = [[] for _ in range(len(multiple_data))]
    if args.full_zstar:
        z_star = [torch.randn((1, num_channels, lengths[i][0]), device=device) for i in range(len(multiple_data))]
    else:
        z_star = [torch.randn((1, 1, lengths[i][0]), device=device).repeat(1, num_channels, 1) for i in range(len(multiple_data))]
    torch.save(z_star, pjoin(args.save_path, 'z_star.pt'))
    reals = [[] for _ in range(len(multiple_data))]
    gt_deltas = [[] for _ in range(len(multiple_data))]
    training_groups = get_group_list(args, len(lengths[0]))
    
    for step in range(len(lengths[0])):
        for i in range(len(multiple_data)):
            length = lengths[i][step]
            motion_data = multiple_data[i]
            reals[i].append(lerp_sample(motion_data,size=length))
            last_real = reals[i][-2] if step > 0 else torch.zeros_like(reals[i][-1])
            amps[i].append(torch.nn.MSELoss()(reals[i][-1], interpolator(last_real, length)) ** 0.5)
            if step == 0 and args.correct_zstar_gen:
                z_star[i] *= amps[i][0]
            gt_deltas[i].append(reals[i][-1] - interpolator(last_real, length))

        create = create_conditional_model if args.conditional_generator and step < args.num_conditional_generator else create_model
        gen, disc, gan_model = create(args, motion_data, evaluation=False, neibor = neibor_list)

        gens.append(gen)
        gans.append(gan_model)

    amps = torch.tensor(amps)
    if not args.requires_noise_amp:
        amps = torch.ones_like(amps)
    torch.save(amps, pjoin(args.save_path, 'amps.pt'))

    last_stage = 0
    for group in training_groups:
        curr_stage = last_stage + len(group)
        group_gan_models = [gans[i] for i in group]
        joint_train(reals, gens[:curr_stage], group_gan_models, lengths, adjusted_label_list, adjusted_fake_label_list,
                    z_star, amps, args, loss_recorder, ConGen)

        for i, gan_model in enumerate(group_gan_models):
            torch.save(gan_model.gen.state_dict(), pjoin(args.save_path, f'gen{group[i]:03d}.pt'))

        last_stage = curr_stage

    end_time = time.time()
    if not args.silent:
        print(f'Training time: {end_time - start_time:.07f}s')


if __name__ == '__main__':
    main()
