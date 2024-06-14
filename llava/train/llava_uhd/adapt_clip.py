import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
from transformers.models.clip.modeling_clip import (CLIPEncoderLayer, CLIPAttention, 
                                                    CLIPVisionEmbeddings,BaseModelOutputWithPooling,
                                                    CLIPEncoder, CLIPVisionTransformer, CLIPPreTrainedModel)
from typing import List, Optional, Tuple, Union
import os
try:
    from slice_logic import get_patch_nums
except:
    from llava.train.llava_uhd.slice_logic import get_patch_nums

#---------------------------------------------------------------------------#
# 用来生成position embedding的层
#---------------------------------------------------------------------------#   
PATCH_SIZE       = 14
PATCH_NUM_WIDTH  = 24
PATCH_NUM_HEIGHT = 24
POSITION_EMBEDDING_LENGTH = 1024
# 196
MAX_PATCHES      = PATCH_NUM_WIDTH * PATCH_NUM_HEIGHT
# 768
TOKEN_LENGTH     = 3 * PATCH_SIZE * PATCH_SIZE
# 224 224
IMAGE_WIDTH      = PATCH_SIZE * PATCH_NUM_WIDTH
IMAGE_HEIGHT     = PATCH_SIZE * PATCH_NUM_HEIGHT

class adapt_CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # def inverse_and_extract(self, pixel_values, patch_heights, patch_widths): # bs, 3, h, w
    #     norm_size = self.image_size // self.patch_size
    #     patch_embeds = []
    #     # print('norm_size:', norm_size)
    #     for img, ph, pw in zip(pixel_values, patch_heights, patch_widths):
    #         # import pdb; pdb.set_trace()
    #         # print(ph, pw, img.sum())
    #         # if ph is None or pw is None:
    #         #     patch_embed = self.patch_embedding(img[None]).flatten(2).transpose(1, 2)
    #         # else:
    #         # 反转图像为正常编码
    #         inversed_img = img.permute(1, 2, 0).reshape(norm_size, 
    #             self.patch_size, norm_size, 
    #             self.patch_size, 3).permute(0, 2, 1, 3, 4).reshape(-1, 
    #             3*self.patch_size**2)[:pw*ph].reshape(ph, pw, self.patch_size, self.patch_size, 3).permute(0, 2, 1, 3, 4).reshape(
    #                 ph * self.patch_size, pw * self.patch_size, 3
    #             ).permute(2, 0, 1)[None] # 1,3,h,w

    #         # import matplotlib.pyplot as plt
    #         # plt.imshow(inversed_img[0].permute(1, 2, 0).float().cpu().numpy())
    #         # plt.savefig('1.png')
    #         # 提取特征
    #         patch_embed = self.patch_embedding(inversed_img).flatten(2).transpose(1, 2)  # 1, c, ph, pw->1,n,c
    #         # padding 0
    #         patch_embed = torch.nn.functional.pad(patch_embed, (0, 0, 0, MAX_PATCHES-ph*pw))

    #         patch_embeds.append(patch_embed)
    #     patch_embeds = torch.cat(patch_embeds)
    #     return patch_embeds

    def get_position_embedding(self,positional_embedding, patch_width_num:int, patch_height_num:int, method = 'bicubic'):
        patch_width_num  = int(patch_width_num)
        patch_height_num  = int(patch_height_num)
        position_embedding = positional_embedding.squeeze(0)
        position_for_class = position_embedding[0, :]  
        #----------------------------------------------------#
        # 插值获得 patch_width_num * patch_height_num 的位置编码
        #----------------------------------------------------#
            #----------------------------------------------------#
            # bicubic 插值
            #----------------------------------------------------#
        if method == 'bicubic':
            position_embedding = position_embedding[1:, :].reshape((PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, POSITION_EMBEDDING_LENGTH))
            position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(1)
            original_dtype = position_embedding.dtype  # 保存原始数据类型

            # 将数据类型更改为 float32 以进行插值
            position_embedding = position_embedding.to(torch.float32)

            # 执行双三次插值
            position_embedding = torch.nn.functional.interpolate(
                position_embedding, size=(int(patch_height_num), int(patch_width_num)),
                mode='bicubic', align_corners=False)

            # 将数据类型改回原来的类型
            position_embedding = position_embedding.to(original_dtype)
            
            position_embedding = position_embedding.squeeze(1).permute(1, 2, 0).reshape(patch_height_num*patch_width_num, POSITION_EMBEDDING_LENGTH)
            #----------------------------------------------------#
            # trilinear 插值
            #----------------------------------------------------#
        elif method == 'trilinear':
            position_embedding = position_embedding[1:, :].reshape((PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, POSITION_EMBEDDING_LENGTH)).unsqueeze(0).unsqueeze(0)
            m = torch.nn.Upsample(( patch_height_num, patch_width_num, POSITION_EMBEDDING_LENGTH), mode = 'trilinear')
            position_embedding = m(position_embedding).squeeze().view(patch_width_num*patch_height_num,POSITION_EMBEDDING_LENGTH)
        
        #-----------------------#
        # 用0补全位置编码缺少的部分
        #-----------------------#
        # position_embedding = torch.nn.functional.pad(position_embedding, (0, 0, 0, MAX_PATCHES-patch_height_num*patch_width_num ))
        # position_embedding = position_embedding.reshape(MAX_PATCHES, POSITION_EMBEDDING_LENGTH)
        # position_embedding = torch.cat((position_for_class.reshape(1,POSITION_EMBEDDING_LENGTH),position_embedding))
        position_embedding = torch.cat([position_for_class.unsqueeze(0), position_embedding], dim=0)
        return position_embedding

    def forward(self,
        pixel_values,
        patch_w,
        patch_h) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        # patch_heights = [img_h // self.patch_size for img_h in origin_image_heights]
        # patch_widths = [img_w // self.patch_size for img_w in origin_image_widths]
        # patch_embeds = self.inverse_and_extract(pixel_values, patch_heights, patch_widths)
        # import pdb; pdb.set_trace()
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # torch.Size([16, 577, 1024])
        processed_position_embedding = self.get_position_embedding(
            self.position_embedding(self.position_ids),
            patch_width_num=patch_w,
            patch_height_num=patch_h).unsqueeze(0)
        
        # torch.cat([
        #     self.get_position_embedding(
        #         self.position_embedding(self.position_ids),
        #         patch_width_num=dim[0],
        #         patch_height_num=dim[1]
        #     ).unsqueeze(0) for dim in list(zip(patch_widths, patch_heights))
        #     # ).unsqueeze(0) for dim in list(zip(origin_image_widths, origin_image_heights))
        # ])
        
        embeddings = embeddings + processed_position_embedding
        return embeddings

class adapt_CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = adapt_CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        patch_w = None,
        patch_h = None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        hidden_states = self.embeddings(pixel_values=pixel_values,
            patch_w=patch_w,
            patch_h=patch_h)
        
        # for patch_h, patch_w, feat in zip(origin_image_heights, origin_image_widths, hidden_states):
        #     valid_num = patch_h * patch_w
        #     if sum(feat.sum(-1) == 0.0) != 576 - valid_num:
        #         print(patch_h, patch_w, feat.sum(-1) == 0.0)

        # patch_heights = [img_h // 14 for img_h in origin_image_heights]
        # patch_widths = [img_w // 14 for img_w in origin_image_widths]

        # sums = hidden_states.sum(dim=-1) # bs, n, c
        # attentionMask = (sums == 0)
        # attentionMask = attentionMask.float()
        
        # attentionMask_temp = torch.ones_like(attentionMask)
        # for i_bs, (ph, pw) in enumerate(zip(patch_heights, patch_widths)):
        #     attentionMask_temp[i_bs][:ph*pw + 1] = 0 # 存在cls token
        # print((attentionMask_temp != attentionMask).sum())

        # attentionMask[attentionMask == 1] = -float('inf')


        # attn_mask = attentionMask.clone()
        # attn_mask[attn_mask == 0] = 1
        # attn_mask = attn_mask.unsqueeze(1) * attn_mask.unsqueeze(-1)
        # import numpy as np
        # np.save('myattn.npy', attn_mask.detach().float().cpu().numpy())
        # import pdb; pdb.set_trace()

        hidden_states = self.pre_layrnorm(hidden_states)
        
        # 添加一个新维度并复制
        # attentionMask = attentionMask.unsqueeze(1).unsqueeze(3).repeat(1, 1, 1, 577).to(torch.bfloat16)
        # attentionMask = attentionMask.unsqueeze(1).unsqueeze(2).repeat(1, 1, 577, 1).to(dtype=hidden_states.dtype)
        # np.save('nowattn.npy', attentionMask.detach().float().cpu().numpy())

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            # attention_mask=attentionMask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class adapt_CLIPVisionModel(CLIPVisionModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = adapt_CLIPVisionTransformer(config)
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        patch_w = None,
        patch_h = None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # import pdb; pdb.set_trace()
        # if pixel_values.shape[0] == 1:
        #     pixel_values = pixel_values.squeeze(0)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_w=patch_w,
            patch_h=patch_h
        )


class adapt_CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = adapt_CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def unfreeze_position_embedding(self):
        self.vision_tower.vision_model.embeddings.requires_grad_(True)
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    # def forward(self, images, origin_image_widths, origin_image_heights, slice_image_widths, slice_image_heights):
    def forward(self, images, patch_images_list):
        # abs feature
        abs_image_features = []
        abs_image_patch_sizes = []
        for image in images: # [3,h,w]
            # print(image.shape)
            # import pdb; pdb.set_trace()
            image = image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
            img_h, img_w = image.shape[-2:]
            patch_h, patch_w = img_h // self.config.patch_size, img_w // self.config.patch_size
            image_forward_out = self.vision_tower(image, output_hidden_states=True, 
                                                  patch_h=patch_h, patch_w=patch_w
                                                  )
            image_feature = self.feature_select(image_forward_out).to(image.dtype)
            # print(image_feature.shape)
            # import pdb; pdb.set_trace()
            abs_image_features.append(image_feature)
            abs_image_patch_sizes.append((patch_h, patch_w))
        
        # slice feature
        slice_image_features = []
        slice_image_patch_sizes = []
        for patches in patch_images_list:
            # print(patches.shape, ind_tokens, len(ind_tokens))
            # import pdb; pdb.set_trace()
            patches = patches.to(device=self.device, dtype=self.dtype)
            slice_img_h, slice_img_w = patches.shape[-2:]
            slice_patch_h, slice_patch_w = slice_img_h // self.config.patch_size, slice_img_w // self.config.patch_size
            image_forward_out = self.vision_tower(patches, output_hidden_states=True, 
                                                  patch_w=slice_patch_w, patch_h=slice_patch_h
                                                  )
            image_feature = self.feature_select(image_forward_out).to(image.dtype)
            # print(image_feature.shape)
            # import pdb; pdb.set_trace()
            slice_image_features.append(image_feature)
            slice_image_patch_sizes.append((slice_patch_h, slice_patch_w))

        return abs_image_features, slice_image_features, abs_image_patch_sizes, slice_image_patch_sizes


        # num_images = images.shape[1] // 3
        # image_features = []
        # split_images = torch.chunk(images, chunks=num_images, dim=1) # b, 3, h, w
            
        # for i, image in enumerate(split_images):
        #     # print(i)
        #     if i == num_images - 1:
        #         image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
        #                                             output_hidden_states=True,
        #                                             origin_image_widths = origin_image_widths,
        #                                             origin_image_heights = origin_image_heights
        #                                             )
        #                                         #   origin_image_widths = slice_w_nums,
        #                                         #   origin_image_heights = slice_h_nums)
        #     else:
        #         image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
        #                                             output_hidden_states=True,
        #                                             origin_image_widths = slice_image_widths,
        #                                             origin_image_heights = slice_image_heights
        #                                             )
        #                                         #   origin_image_widths = abstract_w_nums,
        #                                         #   origin_image_heights = abstract_h_nums)
                
        #     image_feature = self.feature_select(image_forward_out).to(image.dtype)

        #     image_features.append(image_feature)

        # return image_features




        # if images.shape[1] == 24: # b, 24, h,w
        #     image_features = []
        #     split_images = torch.chunk(images, chunks=8, dim=1) # b, 3, h, w
            
        #     slice_w_nums=[]
        #     slice_h_nums=[]
        #     abstract_w_nums=[]
        #     abstract_h_nums=[]
            
        #     for i in range(len(origin_image_widths)): #[b] 
        #         slice_w_num,slice_h_num,abstract_w_num,abstract_h_num = get_patch_nums(origin_image_widths[i],origin_image_heights[i])
        #         slice_w_nums.append(slice_w_num)
        #         slice_h_nums.append(slice_h_num)
        #         abstract_w_nums.append(abstract_w_num)
        #         abstract_h_nums.append(abstract_h_num)
                
                
        #     for i, image in enumerate(split_images):
                
        #         if i == 7:
        #             image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
        #                                               output_hidden_states=True,
        #                                               origin_image_widths = abstract_w_nums,
        #                                               origin_image_heights = abstract_h_nums)
        #                                             #   origin_image_widths = slice_w_nums,
        #                                             #   origin_image_heights = slice_h_nums)
        #         else:
        #             image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
        #                                               output_hidden_states=True,
        #                                               origin_image_widths = slice_w_nums,
        #                                               origin_image_heights = slice_h_nums)
        #                                             #   origin_image_widths = abstract_w_nums,
        #                                             #   origin_image_heights = abstract_h_nums)
                    
        #         image_feature = self.feature_select(image_forward_out).to(image.dtype)

        #         image_features.append(image_feature)

        # else:
        #     image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
        #                                                 output_hidden_states=True,
        #                                                 origin_image_widths = origin_image_widths,
        #                                                 origin_image_heights = origin_image_heights)

        #     image_features = self.feature_select(image_forward_outs).to(images.dtype)


        # return image_features, abstract_w_nums, abstract_h_nums, slice_w_nums, slice_h_nums
        
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def adapt_build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return adapt_CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')