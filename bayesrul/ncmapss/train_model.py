from bayesrul.ncmapss.dataset import NCMAPSSDataModule

from bayesrul.inference.vi_bnn import VI_BNN
from bayesrul.inference.dnn import HomoscedasticDNN, HeteroscedasticDNN
from bayesrul.inference.mc_dropout import MCDropout
from bayesrul.inference.deep_ensemble import DeepEnsemble

from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import schedule

import torch
import argparse

"""
Utility file to run tests, it allows to launch a training of a specific model
with specific options
"""


my_schedule = schedule(
    skip_first=1,
    wait=0,
    warmup=1,
    active=5,
    repeat=1)


if __name__ == "__main__":
    # Launch from root directory : python -m bayesrul.ncmapss.benchmarking
    parser = argparse.ArgumentParser(description='Bayesrul benchmarking')
    parser.add_argument('--data-path',
                    type=str,
                    default='data/ncmapss',
                    metavar='DATA',
                    help='Directory where to find the data')
    parser.add_argument('--out-path',
                    type=str,
                    default='results/ncmapss/',
                    metavar='OUT',
                    help='Directory where to store models and logs')
    parser.add_argument('--model-name',
                    type=str,
                    default='dnn',
                    metavar='NAME',
                    help='Name of this specific run. (default: dnn)',
                    required=True)
    parser.add_argument('--archi',
                    type=str,
                    default='linear',
                    metavar='ARCHI',
                    help='Which model to run. (default: linear)')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--pretrain',
                        type=int,
                        metavar='PRETRAIN',
                        default=0,
                        help='Pretrain the BNN weights for x epoch. (default: 0)')
    parser.add_argument('--bayesian',
                        action='store_true',
                        default=False,
                        help='Wether to train a bayesian model (default: False)')
    parser.add_argument('--last-layer',
                        action='store_true',
                        default=False,
                        help='Having only the last layer as Bayesian (default: False)')
    parser.add_argument('--test',
                        action='store_true',
                        default=False,
                        help='Just run the test phase (default: False)')
    parser.add_argument('--guide',
                    type=str,
                    default='normal',
                    metavar='GUIDE',
                    help='Normal or Radial Autoguide. (default: normal)')
    parser.add_argument('--GPU',
                    type=int,
                    default='results/ncmapss/',
                    metavar='GPU',
                    required=True,
                    help='GPU index (ex: 1)')
 

    args = parser.parse_args()
    

    if args.bayesian:
        hyp = {
                'activation': 'leaky_relu',
                'bias' : True,
                'prior_loc' : 0,
                'prior_scale' : 0.4,
                'likelihood_scale' : 0, # Useless in Heteroskedastic case
                'q_scale' : 0.004,
                'fit_context' : 'null',
                'num_particles' : 1,
                'optimizer': 'adam',
                'lr' : 0.002,
                'last_layer': args.last_layer,
                'pretrain_file' : None,
            }

        data = NCMAPSSDataModule(args.data_path, batch_size=10000)
        module = VI_BNN(args, data, hyp, GPU=args.GPU)
        if not args.test:
            """with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler('results/ncmapss/bayesian/profile/lightning_logs/')
            ) as prof:"""

            module.fit(150)
            #prof.step
        else:
            pass
        module.test()
        module.epistemic_aleatoric_uncertainty(device=torch.device('cpu'))
    else:
        data = NCMAPSSDataModule(args.data_path, batch_size=10000)
        #module = DNN(args, data)
        module = MCDropout(args, data, 0.25)
        if not args.test:
            module.fit(100)
        module.test(device = torch.device(f"cuda:{args.GPU}"))
        module.epistemic_aleatoric_uncertainty(device=torch.device(f"cuda:{args.GPU}"))


