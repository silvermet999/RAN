import argparse
import itertools
import os
import subprocess
import sys

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

import utils
from AAE import AAE_training, AAE_testing, AAE_archi_opt
import mlflow
from mlflow.models import infer_signature
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False
torch.cuda.empty_cache()
torch.manual_seed(0)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_size", default=30, type=int)
    parser.add_argument("--z_dim", default=10, type=int)
    parser.add_argument("--label_dim", default=4, type=int)

    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--numEpochs', default=101, type=int)
    # when the discriminator loss reaches a threshold, we save the AAE state dictionary
    parser.add_argument("--loss_threshold", default=0.5, type=float)
    # define number of interpolations and sample size per interpolation
    # !!!DESIRED DATASIZE = number of interpolations * sample size per interpolation!!!
    parser.add_argument("--n_inter", default=5, type=int) # we set it to 4 when --unaug_dataset = False
    parser.add_argument("--n_samples_per_inter", default=27321, type=int) # we set it to 43313 when --unaug_dataset = False


    # if test: ---train False
    parser.add_argument('--train', action='store_true')
    # unaug = unaugmented dataset = original dataset : if False then augmented dataset
    parser.add_argument("--unaug_dataset", action="store_true")
    parser.add_argument("--dataset_file", default="results/ds.csv")
    # PLEASE USE THE ABSOLUTE PATH IF YOU GET A NO FILE IS FOUND!!!
    parser.add_argument("--save_state_dict", default="results/aae.pth")
    parser.add_argument('--X_ds', default="results/rl_ds1.csv")
    parser.add_argument('--y_ds', default="results/labels.csv")

    return parser.parse_args(args)

if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    # process = subprocess.Popen(
    #     ["mlflow", "server", "--host", "127.0.0.1", "--port", "8080"],
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True
    # )
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    print("using", args.unaug_dataset)
    dataset = utils.dataset(original=args.unaug_dataset, train=args.train)

    # AAE components
    encoder_generator = AAE_archi_opt.EncoderGenerator(args.feature_size, args.z_dim).cuda() if cuda else (
        AAE_archi_opt.EncoderGenerator(args.feature_size, args.z_dim))

    decoder = AAE_archi_opt.Decoder(args.z_dim + args.label_dim, args.feature_size, utils.discrete, utils.continuous,
                                    utils.binary).cuda() if cuda else (
        AAE_archi_opt.Decoder(args.z_dim + args.label_dim, args.feature_size,
                              utils.discrete, utils.continuous, utils.binary))

    discriminator = AAE_archi_opt.Discriminator(args.z_dim, ).cuda() if cuda else (
        AAE_archi_opt.Discriminator(args.z_dim, ))

    # Optimizers
    optimizer_G = SGD(itertools.chain(encoder_generator.parameters(), decoder.parameters()), lr=0.001, momentum=0.9)
    optimizer_D = SGD(discriminator.parameters(), lr=0.001, momentum=0.9)
    # scheduler_G = MultiStepLR(optimizer_D, milestones=[16, 26, 36, 46], gamma=0.1)
    # scheduler_D = MultiStepLR(optimizer_D, milestones=[46, 92], gamma=0.1)

    if args.train:
        train_loader, val_loader = utils.dataset_function(dataset, batch_size_t=args.batch_size_train,
                                                          batch_size_o=args.batch_size_test, train=True)

        best_d_val_loss = args.loss_threshold
        # mlflow.set_experiment("AAE")
        # with mlflow.start_run():
        for epoch in range(args.numEpochs):
            g_loss, d_loss = AAE_training.train_model(train_loader, encoder_generator, decoder, discriminator, args.z_dim,
                                                      optimizer_G, optimizer_D)
            print(f"Epoch {epoch + 1}/{args.numEpochs}, g loss: {g_loss}, d loss: {d_loss}")
            if epoch % 10 == 0:
                g_val, d_val = AAE_training.evaluate_model(val_loader, encoder_generator, decoder, discriminator, args.z_dim)
                # mlflow.log_metric("g val", g_val, step=epoch)
                # mlflow.log_metric("d val", d_val, step=epoch)
                print(f"g loss: {g_val}, d loss: {d_val}")
                if d_val < best_d_val_loss:
                    best_d_val_loss = d_val
                    torch.save({
                                'enc_gen': encoder_generator.state_dict(),
                                'dec': decoder.state_dict(),
                                "disc": discriminator.state_dict(),
                                }, f"{args.save_state_dict}")
            # scheduler_G.step()
            # scheduler_D.step()

            # model_info_gen = mlflow.pytorch.log_model(
            #     pytorch_model = AAE_training.encoder_generator,
            #     artifact_path="mlflow/gen",
            #     input_example=30,
            #     registered_model_name="G_tracking",
            # )
            # model_info_disc = mlflow.pytorch.log_model(
            #     pytorch_model=AAE_training.discriminator,
            #     artifact_path="mlflow/discriminator",
            #     input_example=10,
            #     registered_model_name="D_tracking",
            # )

        d, c, b = AAE_training.sample_runs(decoder, args.n_inter, args.n_samples_per_inter)
        AAE_training.save_features_to_csv(d, c, b, args.dataset_file)

    else:
        test_loader = utils.dataset_function(dataset, batch_size_t=args.batch_size_train,
                                             batch_size_o=args.batch_size_test, train=False)
        g_loss, d_loss = AAE_testing.test_model(test_loader, args.save_state_dict, encoder_generator, decoder, discriminator, args.z_dim)
        print(f"g_loss: {g_loss}, d_loss: {d_loss}")
