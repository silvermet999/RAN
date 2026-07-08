"""-----------------------------------------------import libraries-----------------------------------------------"""
import os
import pandas as pd
from torch.optim import SGD

import utils
from AAE import AAE_archi_opt
import torch
from torch.nn.functional import binary_cross_entropy, one_hot
import itertools


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False
torch.cuda.empty_cache() 
torch.manual_seed(0)

"""------------------------------------------------gradient penalty------------------------------------------------"""
# Gradient penalty for discriminator loss
def gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).cuda() if cuda else torch.rand(real_samples.size(0), 1)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

"""-----------------------------------------------------data gen-----------------------------------------------------"""
# Save samples
def save_features_to_csv(discrete_samples, continuous_samples, binary_samples, dataset_file):
    def dict_to_df(tensor_dict):
        all_data = []
        for sample_idx in range(next(iter(tensor_dict.values())).shape[0]):
            row_data = {}
            for feature_name, tensor in tensor_dict.items():
                if len(tensor.shape) > 2:
                    tensor = tensor.reshape(tensor.shape[0], -1)

                values = tensor[sample_idx].detach().cpu().numpy()
                if len(values.shape) == 0:
                    row_data[f"{feature_name}"] = values.item()
                else:
                    for _, value in enumerate(values):
                        row_data[f"{feature_name}"] = value
            all_data.append(row_data)
        return pd.DataFrame(all_data)

    discrete_df = dict_to_df(discrete_samples)
    continuous_df = dict_to_df(continuous_samples)
    binary_df = dict_to_df(binary_samples)

    combined_df = pd.concat([discrete_df, continuous_df, binary_df], axis=1)
    combined_df.to_csv(f'{dataset_file}', index=False)

    return combined_df

# Interpolation
def interpolate(z1, z2, n_steps=5):
    interpolations = []
    for alpha in torch.linspace(0, 1, n_steps):
        z = z1 * (1 - alpha) + z2 * alpha
        interpolations.append(z)
    return torch.stack(interpolations)

def sample_runs(decoder, n_inter, n_samples_p_inter):
    # data samples with the type filtered
    discrete_samples = {feature: [] for feature in decoder.discrete_features}
    continuous_samples = {feature: [] for feature in decoder.continuous_features}
    binary_samples = {feature: [] for feature in decoder.binary_features}
    decoder.eval()
    with torch.no_grad():
        n_interpolations = n_inter #4 #5
        n_samples_per_interpolation = n_samples_p_inter #43313 #27321
        # interpolation vectors : latent vector dim + label dim = 14
        z1 = torch.randn(n_interpolations, 14).cuda() if cuda else torch.randn(n_interpolations, 14)
        z2 = torch.randn(n_interpolations, 14).cuda() if cuda else torch.randn(n_interpolations, 14)

        for i in range(n_interpolations):
            interpolations = interpolate(z1[i], z2[i], n_samples_per_interpolation)
            discrete_out, continuous_out, binary_out = decoder(interpolations)

            # append and concatenate all samples
            discrete_samples, continuous_samples, binary_samples = utils.types_append(decoder,
                discrete_out, continuous_out, binary_out, discrete_samples, continuous_samples, binary_samples)

        discrete_samples, continuous_samples, binary_samples = utils.type_concat(decoder, discrete_samples,
                                                                                         continuous_samples, binary_samples)

        return discrete_samples, continuous_samples, binary_samples


"""--------------------------------------------------model training--------------------------------------------------"""
def train_model(train_loader, encoder_generator, decoder, discriminator, z_dim, optimizer_G, optimizer_D):
    # Set to train mode
    encoder_generator.train()
    decoder.train()
    discriminator.train()
    g_total = 0.0
    d_total = 0.0
    for i, (X, y) in enumerate(train_loader):
        # valid = matrix of ones that follow the same distribution as the real samples
        valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                   requires_grad=False)
        # fake = matrix of zeros that follow the same distribution as the real samples
        fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                    requires_grad=False)

        real = X.type(torch.FloatTensor).cuda() if cuda else X.type(torch.FloatTensor)
        y = y.type(torch.LongTensor).squeeze().cuda() if cuda else y.type(torch.LongTensor).squeeze()
        # One hot encode the labels
        y = one_hot(y, num_classes=4)

        # targets according to data type
        discrete_targets = {}
        continuous_targets = {}
        binary_targets = {}

        for feature, _ in decoder.discrete_features.items():
            # [:, :4] = the first 3 features are discrete
            discrete_targets[feature] = real[:, :4]

        for feature in decoder.continuous_features:
            # [:, 6:] = starting from the 6th features, they are continuous
            continuous_targets[feature] = real[:, 6:]

        for feature in decoder.binary_features:
            # [:, 4:6] = the 4th and the 5th features are binary
            binary_targets[feature] = real[:, 4:6]

        optimizer_G.zero_grad()
        # samples that follow the same distribution as the real data
        z = torch.rand(real.shape[0], z_dim).cuda() if cuda else torch.rand(real.shape[0], z_dim)
        # encode the latent space
        encoded = encoder_generator(real)
        # join the encoded space with the labels
        dec_input = torch.cat([encoded, y], dim=1)
        discrete_outputs, continuous_outputs, binary_outputs = decoder(dec_input)

        # Generator loss
        g_loss = (0.1 * binary_cross_entropy(discriminator(encoded.detach()), valid) +
                  0.9 * decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                               (discrete_targets, continuous_targets, binary_targets)))

        g_loss.backward()
        optimizer_G.step()

        # Gradient penalty
        gp = gradient_penalty(discriminator, z, encoded)

        optimizer_D.zero_grad()
        # Discriminator loss
        real_loss = binary_cross_entropy(discriminator(z), valid)
        fake_loss = binary_cross_entropy(discriminator(encoded.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss) + 0.2 * gp

        d_loss.backward()
        optimizer_D.step()

        # Total batch losses
        g_total += g_loss.item()
        d_total += d_loss.item()

    g_total_loss = g_total / len(train_loader)
    d_total_loss = d_total / len(train_loader)

    return g_total_loss, d_total_loss


"""-------------------------------------------------model validation-------------------------------------------------"""

def evaluate_model(val_loader, encoder_generator, decoder, discriminator, z_dim):
    # model in eval mode
    encoder_generator.eval()
    decoder.eval()
    discriminator.eval()
    # same steps as training
    # the penalty is not included because we disabled the gradients
    total_g_loss = 0.0
    total_d_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            valid = torch.ones((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones((X.shape[0], 1),
                                                                                                    requires_grad=False)
            fake = torch.zeros((X.shape[0], 1), requires_grad=False).cuda() if cuda else torch.zeros((X.shape[0], 1),
                                                                                                     requires_grad=False)

            real = X.type(torch.FloatTensor).cuda() if cuda else X.type(torch.FloatTensor)
            y = y.type(torch.LongTensor).squeeze().cuda() if cuda else y.type(torch.LongTensor).squeeze()
            y = one_hot(y, num_classes=4)

            discrete_targets = {}
            continuous_targets = {}
            binary_targets = {}

            for feature, _ in decoder.discrete_features.items():
                discrete_targets[feature] = real[:, :3]

            for feature in decoder.continuous_features:
                continuous_targets[feature] = real[:, 5:]

            for feature in decoder.binary_features:
                binary_targets[feature] = real[:, 3:5]

            encoded = encoder_generator(real)

            dec_input = torch.cat([encoded, y], dim=1)
            discrete_outputs, continuous_outputs, binary_outputs = decoder(dec_input)

            g_loss = (0.1 * binary_cross_entropy(discriminator(encoded),
                                                 torch.ones((X.shape[0], 1),
                                                            requires_grad=False).cuda() if cuda else torch.ones(
                                                     (X.shape[0], 1), requires_grad=False)) +
                      0.9 * decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                                 (discrete_targets, continuous_targets, binary_targets)))


            z = torch.rand(real.shape[0], z_dim).cuda() if cuda else torch.rand(real.shape[0], z_dim)

            real_loss = binary_cross_entropy(discriminator(z), valid)
            fake_loss = binary_cross_entropy(discriminator(encoded), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        avg_g_loss = total_g_loss / len(val_loader)
        avg_d_loss = total_d_loss / len(val_loader)
    return avg_g_loss, avg_d_loss


