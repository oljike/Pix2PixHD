import os
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms

from dataset import DatasetFromFolder
from models import LocalEnhancer, GlobalGenerator, MultiscaleDiscriminator, VGGLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Pix2PixHD(nn.Module):
    def __init__(self, num_D, generator, discriminator):
        super(Pix2PixHD, self).__init__()

        self.num_D = num_D
        self.generator = generator
        self.discriminator = discriminator
        self.feat_loss = torch.nn.L1Loss()
        self.vgg_loss = VGGLoss()
        self.criterionGAN = torch.nn.MSELoss()

    def GANloss(self, input_, is_real):

        if is_real:
            label = 1
        else:
            label = 0

        if isinstance(input_[0], list):
            loss = 0.0
            for i in input_:
                pred = i[-1]
                target = torch.Tensor(pred.size()).fill_(label).to(pred.device)
                loss += self.criterionGAN(pred, target)
            return loss
        else:
            target = torch.Tensor(input_[-1].size()).fill_(label).to(input_[-1].device)
            return self.criterionGAN(input_[-1], target)

    def forward(self, label_map, image):
        self.generator.train()

        fake_image = self.generator(label_map.to(device))

        ### Fake Loss
        pred_fake = self.discriminator(torch.cat((label_map, fake_image.detach()), dim=1))
        loss_D_fake = self.GANloss(pred_fake, is_real=False)

        ### Real Loss
        pred_real = self.discriminator(torch.cat((label_map, image.detach()), dim=1))
        loss_D_real = self.GANloss(pred_real, is_real=True)

        ### GAN loss
        pred_fake_GAN = self.discriminator(torch.cat((label_map, fake_image), dim=1))
        loss_GAN = self.GANloss(pred_fake_GAN, is_real=True)

        ### Feature Matching loss
        loss_FM = 0
        D_weights = 1.0 / self.num_D
        for i in range(self.num_D):
            for j in range(len(pred_fake[i]) - 1):
                loss_FM += D_weights * self.feat_loss(pred_fake[i][j], pred_real[i][j].detach()) * 10.0

        loss_VGG = self.vgg_loss(fake_image, image) * 10.0

        return loss_D_fake, loss_D_real, loss_GAN, loss_FM, loss_VGG, fake_image

    def eval(self, label_map):
        self.generator.eval()

        fake_image = self.generator(label_map.to(device))

        return fake_image

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', default=0.00002,
                        help='Initial learning rate')
    parser.add_argument('--beta1', default=0.5,
                        help='Beta value for adam optmizer')
    parser.add_argument('--num_epochs', default=200,
                        help='Number of epochs')
    parser.add_argument('--batch_size', default=1,
                        help='Batch size')
    parser.add_argument('--load_global', default=False,
                        help='Booleab to load global model for transfer learning')
    parser.add_argument('--global_model', default='saved_models/edge_gen_epoch_58.pth',
                        help='Default size of sentences')
    parser.add_argument('--images_dir', default='celeb_pics/imgs',
                        help='Path to images')
    parser.add_argument('--edges_dir', default='celeb_pics/edges',
                        help='Path to edges')
    parser.add_argument('--test_image', default='celeb_pics/test.jpg',
                        help='Path to test image')
    parser.add_argument('--out_dir', default='outputs/train',
                        help='Path to train outputs directory')
    parser.add_argument('--out_test_dir', default='outputs/test',
                        help='Path to train outputs directory')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.load_global:
        generator_global = GlobalGenerator(1, 3)
        generator_global.load_state_dict(torch.load(args.global_model))
        generator = LocalEnhancer(1, 3, generator_global).to(device)
    else:
        generator = GlobalGenerator(1, 3).to(device)

    dataset = DatasetFromFolder(args.images_dir, args.edges_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)

    discriminator = MultiscaleDiscriminator(input_nc=4).to(device)
    model = Pix2PixHD(num_D=3, generator=generator, discriminator=discriminator).to(device)

    optimizer_gen = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))

    landmark_test = Image.open(args.test_image).convert('L')
    landmark_test = landmark_test.resize((256, 128), Image.BICUBIC)
    landmark_test = transforms.ToTensor()(landmark_test)
    landmark_test = landmark_test.to(device)

    epoch = 0
    while epoch < args.num_epochs:
        for en, x in enumerate(dataloader):

            image, label_map = x
            image = image.to(device)
            label_map = label_map.to(device)

            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()

            loss_D_fake, loss_D_real, loss_GAN, loss_FM, loss_VGG, fake_image = model(label_map, image)

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_G = loss_GAN + loss_FM + loss_VGG

            loss_G.backward(retain_graph=True)
            optimizer_gen.step()

            loss_D.backward()
            optimizer_disc.step()

            if en % 1000 == 0 and en!=0:
                print("Discriminator loss", loss_D.item())
                print("Generator loss", loss_G.item())


        img = torchvision.utils.make_grid([image[0].cpu(), label_map[0].repeat(3, 1, 1).cpu(), fake_image[0].cpu()],
                                          nrow=3)
        save_image(img, filename=(os.path.join(args.out_dir, 'fake_img' + str(epoch) + '.jpg')))

        fake_test = model.eval(landmark_test.unsqueeze(0))
        save_image(fake_test.squeeze(0),
                   filename=(os.path.join(args.out_test_dir, 'fake_test' + str(epoch) + '.jpg')))

        print("Epoch %s:  Generator loss %s" %(epoch, str(loss_G.item())))
        epoch += 1