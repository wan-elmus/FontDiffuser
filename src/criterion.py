import torch
import torch.nn as nn
import torchvision 


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(weights='IMAGENET1K_V1')

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, f'enc_{i+1}').parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i+1}')
            results.append(func(results[-1]))
        return results[1:]


class ContentPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG = VGG16()

    def calculate_loss(self, generated_images, target_images, device):
        self.VGG = self.VGG.to(device)

        generated_features = self.VGG(generated_images)
        target_features = self.VGG(target_images)

        perceptual_loss = 0
        perceptual_loss += torch.mean((target_features[0] - generated_features[0]) ** 2)
        perceptual_loss += torch.mean((target_features[1] - generated_features[1]) ** 2)
        perceptual_loss += torch.mean((target_features[2] - generated_features[2]) ** 2)
        perceptual_loss /= 3
        return perceptual_loss


class ShadingTextureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_gram_matrix(self, features):
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (channels * height * width)

    def calculate_loss(self, generated_images, target_images, device):
        vgg = VGG16().to(device)
        
        generated_features = vgg(generated_images)
        target_features = vgg(target_images)

        shading_loss = 0
        for gen_feat, tgt_feat in zip(generated_features, target_features):
            gen_gram = self.compute_gram_matrix(gen_feat)
            tgt_gram = self.compute_gram_matrix(tgt_feat)
            shading_loss += torch.mean((gen_gram - tgt_gram) ** 2)
        shading_loss /= len(generated_features)
        
        return shading_loss


class BackgroundPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.VGG = VGG16()

    def calculate_loss(self, generated_images, target_images, device):
        self.VGG = self.VGG.to(device)

        generated_features = self.VGG(generated_images)
        target_features = self.VGG(target_images)

        perceptual_loss = 0
        perceptual_loss += torch.mean((target_features[0] - generated_features[0]) ** 2)
        perceptual_loss += torch.mean((target_features[1] - generated_features[1]) ** 2)
        perceptual_loss += torch.mean((target_features[2] - generated_features[2]) ** 2)
        perceptual_loss /= 3
        return perceptual_loss