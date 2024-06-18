from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
# from ..llava_arch import  LlavaMetaForCausalLM
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

try:
    from adapt_clip import adapt_build_vision_tower
    from vision_projector import build_vision_projector
except:
    from llava.train.llava_uhd.adapt_clip import adapt_build_vision_tower
    from llava.train.llava_uhd.vision_projector import build_vision_projector

from transformers.generation.utils import GenerateOutput

NEWLINE_TOKEN = 13
DOT_TOKEN = 29892

from abc import ABC, abstractmethod

#_____MY_DEBUG____belong to llava_arch.py

class adapt_LlavaMetaModel:

    def __init__(self, config):
        super(adapt_LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = adapt_build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = adapt_build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
   
        
class adapt_LlavaMetaForCausalLM(ABC):
    
    @abstractmethod
    def get_model(self):
        pass
    
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    # def encode_images(self, images, origin_image_widths, origin_image_heights, 
                            # slice_image_widths, slice_image_heights):
    def encode_images(self, images, patch_images_list, ind_tokens_list):
        # image_features, abstract_w_nums, \
        #     abstract_h_nums, slice_w_nums, slice_h_nums = \
        # image_features = \
        #         self.get_model().get_vision_tower()(images, origin_image_widths, 
        #                                             origin_image_heights, slice_image_widths, 
        #                                             slice_image_heights)
        
        abs_image_features, slice_image_features, \
            abs_image_patch_sizes, slice_image_patch_sizes = self.get_model().get_vision_tower()(images, patch_images_list)
        assert len(abs_image_features) == len(slice_image_features) \
                == len(abs_image_patch_sizes) == len(slice_image_patch_sizes) \
                == len(ind_tokens_list), "the length of abs and patches must be the same"
        # abstract_w_nums, abstract_h_nums = [w // 14 for w in origin_image_widths], [h // 14 for h in origin_image_heights]
        # slice_w_nums, slice_h_nums = [w // 14 for w in slice_image_widths], [h // 14 for h in slice_image_heights]
        # import pdb; pdb.set_trace()
        image_features_list = []
        for abs_image_feature, slice_image_feature, abs_image_patch_size, slice_image_patch_size, ind_tokens in \
            zip(abs_image_features, slice_image_features, abs_image_patch_sizes, slice_image_patch_sizes, ind_tokens_list):
            abs_resampled_feature = self.get_model().mm_projector(abs_image_feature, tgt_size=abs_image_patch_size)
            slice_resampled_feature = self.get_model().mm_projector(slice_image_feature, tgt_size=slice_image_patch_size)
            # print('after resampler but no select', abs_resampled_feature.shape, slice_resampled_feature.shape, len(ind_tokens))
            # import pdb; pdb.set_trace()
            if len(ind_tokens) == 0: # 没有切片
                resampled_image_features = torch.cat([slice_resampled_feature[0:0], abs_resampled_feature], dim=0)
            else: # 有切片 # 由于padding了，所以需要只取前n个
                resampled_image_features = torch.cat([slice_resampled_feature[:len(ind_tokens)], abs_resampled_feature], dim=0)
            # print(torch.isnan(resampled_image_features).sum())
            image_features_list.append(resampled_image_features)
            # print('resampler', resampled_image_features.shape)
            # import pdb; pdb.set_trace()
        return image_features_list

        # if isinstance(image_features,list):
        #     image_features_list = []
        #     for i_slice, image_feature in enumerate(image_features):
        #         # resampler_attn_mask = torch.zeros(len(abstract_w_nums), self.get_model().mm_projector.num_heads, self.get_model().mm_projector.num_queries, image_feature.shape[1]).to(device=image_feature.device, dtype=torch.float32) # bs, num_heads, num_q, num_k
        #         if i_slice == len(image_features) - 1: # 处理缩略
        #             num_valid_tokens = [abs_w * abs_h for abs_w, abs_h in zip(abstract_w_nums, abstract_h_nums)]
        #             patch_w_nums = abstract_w_nums
        #             patch_h_nums = abstract_h_nums
        #         else: # 处理切片
        #             num_valid_tokens = [slice_w * slice_h for slice_w, slice_h in zip(slice_w_nums, slice_h_nums)]
        #             patch_w_nums = slice_w_nums
        #             patch_h_nums = slice_h_nums

        #         # for i, num in enumerate(num_valid_tokens):
        #         #     resampler_attn_mask[i][:, :, num:] = float('-inf')
        #         # resampler_attn_mask = resampler_attn_mask.reshape(-1, self.get_model().mm_projector.num_queries, image_feature.shape[1]).to(image_feature.dtype)
        #         # image_features_list.append(self.get_model().mm_projector(image_feature, attn_mask=resampler_attn_mask))
        #         # resampled_features = self.get_model().mm_projector(image_feature)
        #         # import pdb; pdb.set_trace()
        #         resampled_features = []
        #         for feature, valid_num, patch_h, patch_w in zip(image_feature, num_valid_tokens, patch_h_nums, patch_w_nums):
        #             # print(feature.shape, valid_num, patch_h, patch_w)
        #             # import pdb; pdb.set_trace()
        #             resampled_feature = self.get_model().mm_projector(feature[:valid_num][None], tgt_size=(patch_h, patch_w))
        #             resampled_features.append(resampled_feature)
        #         resampled_features = torch.cat(resampled_features, dim=0)
        #         image_features_list.append(resampled_features)

        #     image_features = torch.concat(tuple(image_features_list), dim=1)
        # else:
        #     image_features = self.get_model().mm_projector(image_features)
            
        # return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, 
        images, 
        # origin_image_widths, origin_image_heights, slice_image_widths, slice_image_heights, num_images_list, 
        patch_images_list,
        ind_tokens_list
    ):
        # import pdb; pdb.set_trace()
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # assert False
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # import pdb; pdb.set_trace()
        # image_features = self.encode_images(images, origin_image_widths, origin_image_heights, 
        #                                     slice_image_widths, slice_image_heights).to(self.device)
        image_features = self.encode_images(images, patch_images_list, ind_tokens_list) #.to(self.device)

        # import pdb; pdb.set_trace()
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # import pdb
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            
            # print(num_images_list, cur_image_idx, num_images, input_ids)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # print(cur_input_embeds_1.shape, cur_image_features.shape,  cur_image_features[0:0].shape)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0, 0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []

            # print(num_images_list, cur_image_idx, num_images, input_ids)
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_ind_tokens = ind_tokens_list[cur_image_idx]
                    cur_image_features = image_features[cur_image_idx] #n, 64, c
                    cur_image_idx += 1
                    num_slices = len(cur_ind_tokens)
                    assert num_slices+1 == cur_image_features.shape[0]
                    
                    cur_abs_image_features = cur_image_features[-1]
                    cur_slice_image_features = cur_image_features[:-1]

                    if num_slices == 0: #没有切片
                        cur_image_features = cur_abs_image_features
                    else:
                        # ind_token切片
                        # print(self.get_model().embed_tokens.weight.requires_grad)
                        cur_ind_tokens_embeds = self.get_model().embed_tokens(
                                    torch.as_tensor(cur_ind_tokens,  # \n , -> 13, 1919
                                                    dtype=torch.long, 
                                                    device=cur_image_features.device))
                        # concat slice+ind tokens
                        temp_cur_image_features = []
                        for slice_image_features, ind_token_embeds in zip(cur_slice_image_features, cur_ind_tokens_embeds):
                            slice_image_features = torch.cat([slice_image_features, ind_token_embeds[None]], dim=0)
                            temp_cur_image_features.append(slice_image_features)
                        temp_cur_image_features.append(cur_abs_image_features)
                        cur_image_features = torch.cat(temp_cur_image_features, dim=0)

                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), 
                                                     IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    
                    # # import pdb; pdb.set_trace()
                    # num_slices = num_images_list[cur_image_idx] # 包括切片+原图 （>1 为切片+原图数量， ==1为原图）
                    # cur_image_features = image_features[cur_image_idx]
                    # cur_ind_tokens = ind_tokens[cur_image_idx] # ind_token 切片数量 == num_slices - 1，如果无切片，这个list=[]
                    # cur_image_idx += 1
                    
                    # num_slice_ind_tokens = int((torch.as_tensor(cur_ind_tokens) != 0).sum())
                    # assert num_slice_ind_tokens == (num_slices - 1)
                    # # 特征切片
                    # cur_image_features = cur_image_features[-num_slices * self.get_model().mm_projector.num_queries:]
                    # cur_all_slice_image_features = cur_image_features.chunk(chunks=num_slices, dim=0)
                    # cur_abs_image_features = cur_all_slice_image_features[-1]
                    # cur_slice_image_features = cur_all_slice_image_features[:-1]

                    # # ind_token切片
                    # cur_ind_tokens_embeds = self.get_model().embed_tokens(
                    #             torch.as_tensor(cur_ind_tokens,  # \n , -> 13, 1919
                    #                             dtype=torch.long, 
                    #                             device=image_features.device)).detach()
                    # cur_ind_tokens_embeds = cur_ind_tokens_embeds[-num_slice_ind_tokens:]

                    # if len(cur_slice_image_features) == 0: # 只有一个图被切出来
                    #     cur_image_features = cur_abs_image_features
                    #     # print(cur_image_features.shape, cur_ind_tokens_embeds.shape, len(cur_slice_image_features), 
                    #     #       len(cur_all_slice_image_features), num_slices)
                    # else:
                    #     temp_cur_image_features = []
                    #     for slice_image_features, ind_token_embeds in zip(cur_slice_image_features, cur_ind_tokens_embeds):
                    #         # import pdb; pdb.set_trace()
                    #         slice_image_features = torch.cat([slice_image_features, ind_token_embeds[None]], dim=0)
                    #         temp_cur_image_features.append(slice_image_features)
                    #     # for slice_image_features in cur_slice_image_features:
                    #     #     temp_cur_image_features.append(slice_image_features)
                    #     temp_cur_image_features.append(cur_abs_image_features)
                    #     cur_image_features = torch.cat(temp_cur_image_features, dim=0)
                    #     # print(cur_image_features.shape, cur_ind_tokens_embeds.shape, len(cur_slice_image_features), 
                    #     #       len(cur_all_slice_image_features), num_slices)
                    # cur_new_input_embeds.append(cur_image_features)
                    # cur_new_labels.append(torch.full((cur_image_features.shape[0],), 
                    #                                  IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        tokenizer_model_max_length = 4096

        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

#_____MY_DEBUG____belong to llava_llama.py
class LlavaConfig(LlamaConfig):
    model_type = "llava_uhd"

class adapt_LlavaLlamaModel(adapt_LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(adapt_LlavaLlamaModel, self).__init__(config)

class adapt_LlavaLlamaForCausalLM(LlamaForCausalLM, adapt_LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = adapt_LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        # origin_image_widths=None,
        # origin_image_heights=None,
        # slice_image_widths=None,
        # slice_image_heights=None,
        # num_images_list=None,
        image_sizes=None, # no use
        patch_images_list=None,
        ind_tokens_list=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                # origin_image_widths,
                # origin_image_heights,
                # slice_image_widths,
                # slice_image_heights,
                # num_images_list,
                patch_images_list,
                ind_tokens_list
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        patch_images_list: Optional[torch.Tensor] = None,
        ind_tokens_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                # image_sizes=image_sizes
                patch_images_list,
                ind_tokens_list
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        # origin_image_widths = kwargs.pop("origin_image_widths", None)
        # origin_image_heights = kwargs.pop("origin_image_heights", None)
        # slice_image_widths = kwargs.pop("slice_image_widths", None)
        # slice_image_heights = kwargs.pop("slice_image_heights", None)
        # num_images_list = kwargs.pop("num_images_list", None)
        ind_tokens_list = kwargs.pop("ind_tokens_list", None)
        patch_images_list = kwargs.pop("patch_images_list", None)
        # print(inputs_embeds.shape)
        # import pdb; pdb.set_trace()
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if image_sizes is not None:
            _inputs['image_sizes'] = image_sizes
        # if origin_image_widths is not None:
        #     _inputs['origin_image_widths'] = origin_image_widths
        # if origin_image_heights is not None:
        #     _inputs['origin_image_heights'] = origin_image_heights
        # if slice_image_widths is not None:
        #     _inputs['slice_image_widths'] = slice_image_widths
        # if slice_image_heights is not None:
        #     _inputs['slice_image_heights'] = slice_image_heights
        # if num_images_list is not None:
        #     _inputs['num_images_list'] = num_images_list
        if ind_tokens_list is not None:
            _inputs['ind_tokens_list'] = ind_tokens_list
        if patch_images_list is not None:
            _inputs['patch_images_list'] = patch_images_list
        return _inputs

AutoConfig.register("llava_uhd", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, adapt_LlavaLlamaForCausalLM)
