import math
import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class FontDiffuserModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
        shading_encoder,
        background_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.shading_encoder = shading_encoder
        self.background_encoder = background_encoder
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        style_images,
        content_images,
        shading_images,
        background_images,
        content_encoder_downsample_size,
    ):
        style_img_feature, _, _ = self.style_encoder(style_images)
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
    
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        shading_img_feature = self.shading_encoder(shading_images)
        background_img_feature = self.background_encoder(background_images)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
            shading_img_feature,
            background_img_feature
        ]

        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]
        
        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self, 
        unet, 
        style_encoder,
        content_encoder,
        shading_encoder,
        background_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.shading_encoder = shading_encoder
        self.background_encoder = background_encoder
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        cond,
        content_encoder_downsample_size,
        version,
    ):
        content_images = cond[0]
        style_images = cond[1]
        shading_images = cond[2]
        background_images = cond[3]

        style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        shading_img_feature = self.shading_encoder(shading_images)
        background_img_feature = self.background_encoder(background_images)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
            shading_img_feature,
            background_img_feature
        ]

        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        
        return noise_pred