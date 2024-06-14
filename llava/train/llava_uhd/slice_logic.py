import math
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
from torchvision.transforms import ToTensor, ToPILImage
import torch

#-------------------------------------------------------#
#  预处理图像
#-------------------------------------------------------#
PATCH_SIZE       = 14
PATCH_NUM_WIDTH  = 24
PATCH_NUM_HEIGHT = 24
POSITION_EMBEDDING_LENGTH = 1024
# 576
MAX_PATCHES      = PATCH_NUM_WIDTH * PATCH_NUM_HEIGHT
# 
TOKEN_LENGTH     = 3 * PATCH_SIZE * PATCH_SIZE
# 336 336
IMAGE_WIDTH      = PATCH_SIZE * PATCH_NUM_WIDTH
IMAGE_HEIGHT     = PATCH_SIZE * PATCH_NUM_HEIGHT

NEWLINE_TOKEN = 13 # '\n'
DOT_TOKEN = 29892  #  ','
def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    Utiliy function to extract patches from a given image tensor. Returns a tensor of shape (1, `patch_height`,
    `patch_width`, `num_channels`x `patch_height` x `patch_width`)

    Args:
        image_tensor (torch.Tensor):
            The image tensor to extract patches from.
        patch_height (int):
            The height of the patches to extract.
        patch_width (int):
            The width of the patches to extract.
    """

    image_tensor = image_tensor.unsqueeze(0) # 1, 3, h, w
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width)) # 1, 14*14*3, ph*pw
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1) # 1,3,14,14,ph*pw
    patches = patches.permute(0, 4, 2, 3, 1).reshape( # 1,phxpw,14,14,3
        image_tensor.size(2) // patch_height, 
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches.unsqueeze(0)

# 用于计算adapt需要输入图片的大小
def adapt_size(originHeight:int,originWeight:int, \
            patchHeight:int = PATCH_SIZE,patchWidth:int = PATCH_SIZE, \
            maxPatches:int = MAX_PATCHES):
    ### 用于计算adapt的图片大小
    # 参数说明 
    # originHeight:              原图高度
    # originWidth:               原图宽度
    # patchHeight:               patch高度
    # patchWidth:                patch宽度
    # maxPatches:                patch数目上限
    # 返回值说明:
    # resized_height:            插值后图片高度
    # resized_width:             插值后图片宽度
    # resized_patch_height_num:  插值后图片垂直patch数目
    # resized_patch_width_num:   插值后图片水平patch数目
    scale = math.sqrt(maxPatches * (patchHeight / originHeight) * (patchWidth / originWeight))
    resized_patch_height_num = max(min(math.floor(scale * originHeight / patchHeight), maxPatches), 1)
    resized_patch_width_num = max(min(math.floor(scale * originWeight / patchWidth), maxPatches), 1)
    resized_height = max(resized_patch_height_num * PATCH_SIZE, 1)
    resized_width = max(resized_patch_width_num * PATCH_SIZE, 1)
    return resized_height, resized_width, resized_patch_height_num, resized_patch_width_num

def cal_num_of_slices(origin_image_width, origin_image_height):
    scale = origin_image_width*origin_image_height/(IMAGE_WIDTH*IMAGE_HEIGHT)  
    scale = math.ceil(scale)
    if scale > 6:
        scale = 6
    def factorize(n):
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append((i/(n/i), i, n // i))
        return factors
    numbers = [1, 2, 3, 4, 5, 6, 7]
    factor_dict = {}
    for num in numbers:
        factor_dict[num] = factorize(num)
    log_origin_ratio = math.log(origin_image_width/origin_image_height)
    available_ratios = []
    if scale<=2:
        available_ratios = factor_dict[scale] + factor_dict[scale + 1]
    else :
        available_ratios = factor_dict[scale-1] + factor_dict[scale]+factor_dict[scale+1]
    min_dif = 1000 
    best_w = 0
    best_h = 0
    for (r,w_slice,h_slice) in available_ratios:
        log_r = math.log(r)
        if min_dif > abs(log_r - log_origin_ratio):
            min_dif = abs(log_r - log_origin_ratio)
            best_w = w_slice
            best_h = h_slice
    
    return best_w,best_h
# 做图片切片     
def get_patch_nums(origin_image_width, origin_image_height):
    # 输入原图的尺寸
    # 返回：
    # slice_w_num 切片的w方向有多少个patch
    # slice_h_num 切片的h方向有多少个patch
    # abstract_w_num 原图的w方向有多少个patch
    # abstract_h_num 原图的h方向有多少个patch
    
    best_w, best_h = cal_num_of_slices(origin_image_width,origin_image_height)
    slice_width = origin_image_width//best_w
    slice_height = origin_image_height//best_h
    _,_,slice_h_num,slice_w_num = adapt_size(slice_height,slice_width)
    _,_,abstract_h_num,abstract_w_num = adapt_size(origin_image_height,origin_image_width)

    return slice_w_num,slice_h_num,abstract_w_num,abstract_h_num

def slice_image(image):
    
    # slice the image according to our princeple
    # return an array of slices
    
    origin_image_width  = image.size[0]
    origin_image_height = image.size[1]

    best_w, best_h = cal_num_of_slices(origin_image_width=origin_image_width,origin_image_height=origin_image_height)
    
    slices = []
    ind_tokens = []
    for j in range(best_h):
        for i in range(best_w):
            
            box = (i * origin_image_width//best_w, j * origin_image_height//best_h, (i + 1) * origin_image_width//best_w, (j + 1) * origin_image_height//best_h)
         
            region = image.crop(box).convert("RGB")
            slices.append(region)

            if i == best_w - 1:
                ind_tokens.append(NEWLINE_TOKEN)
            else:
                ind_tokens.append(DOT_TOKEN)
          
    return slices, ind_tokens



def slice_image_2x2(image):
    
    # slice the image according to our princeple
    # return an array of slices
    
    origin_image_width  = image.size[0]
    origin_image_height = image.size[1]

    best_w, best_h = cal_num_of_slices(origin_image_width=origin_image_width,origin_image_height=origin_image_height)
    
    best_w = best_h = 2

    slices = []
    ind_tokens = []
    for j in range(best_h):
        for i in range(best_w):
            
            box = (i * origin_image_width//best_w, j * origin_image_height//best_h, (i + 1) * origin_image_width//best_w, (j + 1) * origin_image_height//best_h)
         
            region = image.crop(box).convert("RGB")
            slices.append(region)

            if i == best_w - 1:
                ind_tokens.append(NEWLINE_TOKEN)
            else:
                ind_tokens.append(DOT_TOKEN)
          
    return slices, ind_tokens

def slice_image_3x3(image):
    
    # slice the image according to our princeple
    # return an array of slices
    
    origin_image_width  = image.size[0]
    origin_image_height = image.size[1]

    best_w, best_h = cal_num_of_slices(origin_image_width=origin_image_width,origin_image_height=origin_image_height)
    
    best_w = best_h = 3

    slices = []
    ind_tokens = []
    for j in range(best_h):
        for i in range(best_w):
            
            box = (i * origin_image_width//best_w, j * origin_image_height//best_h, (i + 1) * origin_image_width//best_w, (j + 1) * origin_image_height//best_h)
         
            region = image.crop(box).convert("RGB")
            slices.append(region)

            if i == best_w - 1:
                ind_tokens.append(NEWLINE_TOKEN)
            else:
                ind_tokens.append(DOT_TOKEN)
          
    return slices, ind_tokens

def process_image(image, ori_image=False, fix_size=False):
    origin_image_width  = image.size[0]
    origin_image_height = image.size[1]

    image = image.convert("RGB")
    if fix_size:
        # slices, ind_tokens = slice_image_2x2(image)
        # slices, ind_tokens = slice_image_3x3(image)
        slices = [image]
    else:
        slices, ind_tokens = slice_image(image)
    
    # 计算resize之后的图片大小
    resized_height, resized_width, resized_patch_height, resized_patch_width = \
    adapt_size(origin_image_height,origin_image_width)
    
    
    if len(slices) == 1:
        image = slices[0]
        image_w = image.size[0]
        image_h = image.size[1]
        resized_height, resized_width, resized_patch_height, resized_patch_width = \
        adapt_size(image_h,image_w)     
        
        image = ToTensor()(image)
    
        image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        ori_im = image #.permute(1, 2, 0)
        # 需要mask的patch数
        num_patches_to_pad = MAX_PATCHES - resized_patch_height*resized_patch_width
        # raprint("mask: ",num_patches_to_pad)
        # 切割resize好的图片
        image = torch_extract_patches(image,PATCH_SIZE, PATCH_SIZE)
        image = image.reshape([resized_patch_width*resized_patch_height,TOKEN_LENGTH])
        if ori_image:
            return [ori_im], ind_tokens
        # else:
        # 用0补全需要mask的图片部分
        image = torch.nn.functional.pad(image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
        image = image.reshape(PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3).permute(2, 0 ,1)
        # print(image)
        return [image], []
    
    else:
        images = []
        resized_patch_widths = []
        resized_patch_heights = []
        slices.append(image)
        for image in slices:
            image = ToTensor()(image)
    
            image = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(resized_height, resized_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)
            # 需要mask的patch数
            ori_im = image #.permute(1, 2, 0)
            num_patches_to_pad = MAX_PATCHES - resized_patch_height*resized_patch_width
            # raprint("mask: ",num_patches_to_pad)
            # 切割resize好的图片
            image = torch_extract_patches(image,PATCH_SIZE, PATCH_SIZE)
            image = image.reshape([resized_patch_width*resized_patch_height,TOKEN_LENGTH])

            if ori_image:
            #     print(ori_im.shape)
                images.append(ori_im)
            #     resized_patch_widths.append(resized_patch_width)
            #     resized_patch_heights.append(resized_patch_height)
            # else:
            #     # 用0补全需要mask的图片部分
            #     image = torch.nn.functional.pad(image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
            #     image = image.reshape(PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3).permute(2, 0 ,1)
            
            #     # print(image)
            #     images.append(image)
            #     resized_patch_widths.append(resized_patch_width)
            #     resized_patch_heights.append(resized_patch_height)

            # 用0补全需要mask的图片部分
            # image = torch.nn.functional.pad(image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
            # image = image.reshape(PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3).permute(2, 0 ,1)
        
            # # print(image)
            # images.append(image)
            # resized_patch_widths.append(resized_patch_width)
            # resized_patch_heights.append(resized_patch_height)
        return images, ind_tokens

def process_image_(image):
    origin_image_width  = image.size[0]
    origin_image_height = image.size[1]
    image = image.convert("RGB")

    source_image, patches, best_grid = slice_image_minicpm(
        image=image, max_slice_nums=6, scale_resolution=336, patch_size=PATCH_SIZE, never_split=False)
    abs_width, abs_height = source_image.size
    source_image = ToTensor()(source_image)
    abs_patch_width, abs_patch_height = abs_width // PATCH_SIZE, abs_height // PATCH_SIZE
    num_patches_to_pad = MAX_PATCHES - abs_patch_width * abs_patch_height
    abs_image = torch_extract_patches(source_image, PATCH_SIZE, PATCH_SIZE)
    abs_image = abs_image.reshape([abs_patch_height * abs_patch_width, TOKEN_LENGTH])
    # 用0补全需要mask的图片部分
    abs_image = torch.nn.functional.pad(abs_image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
    abs_image = abs_image.reshape(PATCH_NUM_WIDTH, 
        PATCH_NUM_HEIGHT, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3).permute(2, 0, 1)
    abs_image = ToPILImage()(abs_image)
    
    # 只有一片
    if len(patches) == 0:
        return [abs_image], [], abs_width, abs_height, 336, 336 # 如果image number =1 并且split是有24，24的，没用不用管，为了方便forward
    # 有多片
    else:
        patches_abs_images = []
        patches = [item for sublist in patches for item in sublist]
        for patch in patches:
            slice_width, slice_height = patch.size
            patch = ToTensor()(patch) # 3, h, w
            slice_patch_width, slice_patch_height = slice_width // PATCH_SIZE, slice_height // PATCH_SIZE
            num_patches_to_pad = MAX_PATCHES - slice_patch_width * slice_patch_height
            slice_image = torch_extract_patches(patch, PATCH_SIZE, PATCH_SIZE)
            slice_image = slice_image.reshape([slice_patch_height * slice_patch_width, TOKEN_LENGTH]) # ph*pw, 14*14*3
            slice_image = torch.nn.functional.pad(slice_image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
            slice_image = slice_image.reshape(PATCH_NUM_HEIGHT, 
                PATCH_NUM_WIDTH, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3).permute(2, 0, 1)
            slice_image = ToPILImage()(slice_image) # 3, h, w -> h, w, 3
            patches_abs_images.append(slice_image)
        patches_abs_images.append(abs_image)

        ind_tokens = []
        best_w, best_h = best_grid
        for j in range(best_h):
            for i in range(best_w):
                if i == best_w - 1:
                    ind_tokens.append(NEWLINE_TOKEN)
                else:
                    ind_tokens.append(DOT_TOKEN)

        return patches_abs_images, ind_tokens, abs_width, abs_height, slice_width, slice_height

def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches

def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size
    
def ensure_divide(length, patch_size):
    # return max(round(length / patch_size) * patch_size, patch_size)
    return max(math.floor(length / patch_size) * patch_size, patch_size)

def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height # width=672 height=448 r= 1.5
        height = int(scale_resolution / math.sqrt(r)) # scale_resolution=336 / r**0.5  274.3428511917
        width = int(height * r) # 411.5142767876
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)

def slice_image_minicpm(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)
    
    ind_tokens = []
    if best_grid is None:
        return source_image, patches, best_grid, ind_tokens
    else:
        # flatten the patches
        patches = [item for sublist in patches for item in sublist]
        # calculate ind_token layout
        for j in range(best_grid[1]):
            for i in range(best_grid[0]):
                if i != best_grid[0] - 1:
                    ind_tokens.append(DOT_TOKEN)
                else:
                    ind_tokens.append(NEWLINE_TOKEN)

        return source_image, patches, best_grid, ind_tokens
    # source_image: reshape的pil原图
    # patches: 如果没有split，patches=[]，否则为二维list按best grid排列
    # best: 如果有切片，best grid为切片布局如2x3，否则为None
    # ind_token, 有切片就是,和\n的排序。没切片就是空[]


# # # img = Image.open("/home/xuruyi/myLLaVa/883700e3366b775c93315373510e7e7.png")
# img_dir = '/home/guozonghao/LLaVA-UHD/playground/data/LLaVA-Pretrain/images/00152/001529327.jpg'
# img_name = img_dir.split('/')[-1]
# img = Image.open(img_dir)
# img.save('./' + img_name)
# import numpy as np
# img_np = np.array(img)

# image_processor = CLIPImageProcessor.from_pretrained('/home/guozonghao/pretrained_models/clip-vit-large-patch14-336')
# image = image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
# image_1 = image_processor.preprocess(img, do_resize=False, 
#                                      do_center_crop=False,
#                                      do_rescale=True,
#                                      do_normalize=True,
#                                      return_tensors='pt')['pixel_values'][0]
# imagex = image_processor.preprocess([img] * 5, do_resize=False, 
#                                      do_center_crop=False,
#                                      do_rescale=True,
#                                      do_normalize=True,
#                                      return_tensors='pt')['pixel_values']
# to_pil = ToPILImage()
# im_ = to_pil(image_1)
# im_.save(f'{img_name}_norm.png')
# # import pdb; pdb.set_trace()
# # exit()

# print(img.size)
# print(img_np.shape)
# # re_size = (672, 1008)
# # re_size = (336 * 2, 1008)
# re_size = (672, 336)
# img = img.resize(re_size)
# img.save('./' + img_name.split('.')[0] + '_' + str(re_size[0]) + '_' + str(re_size[1]) + '_.' + img_name.split('.')[1])

# # images, ind_tokens = process_image(img, ori_image=True)
# # print(ind_tokens, len(ind_tokens), len(images))
# # for i in range(len(images)):
# #     img_ = images[i]
# #     to_pil = ToPILImage()
# #     img_ = to_pil(img_)
# #     print(img_.size)
# #     img_.save(f"{img_name.split('.')[0]}_{i}.png")


# # images, ind_tokens, abs_width, abs_height, slice_width, slice_height = process_image_(img)
# # print([im.size for im in images], ind_tokens, abs_width, abs_height, slice_width, slice_height)
# # # xx = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
# # # print(xx.shape)

# # # for i in range(len(xx)):
# #     # img_ = xx[i]
# #     # to_pil = ToPILImage()
# #     # img_ = to_pil(img_)

# # for i in range(len(images)):
# #     img_ = images[i]
# #     # 变成原来的图像
# #     print(np.array(img_).shape) # h, w, 3
# #     img_np_ = np.array(img_).reshape(24, 14, 24, 14, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 14*14*3)
# #     # patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width)) # 1, 14*14*3, ph*pw
# #     # patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1) # 1,3,14,14,ph*pw->1,ph*pw,14,14,3
# #     # patches = patches.permute(0, 4, 2, 3, 1).reshape( # 1,phxpw,14,14,3
# #     #     image_tensor.size(2) // patch_height, 
# #     #     image_tensor.size(3) // patch_width,
# #     #     image_tensor.size(1) * patch_height * patch_width,
# #     # )


# #     if i == len(images) - 1:
# #         img_np_ = img_np_[:abs_width*abs_height//14//14].reshape(abs_height//14, abs_width //14, 14, 14, 3).transpose(0, 2, 1, 3, 4).reshape(abs_height, abs_width, 3) #.transpose(1, 0, 2)
# #     else:
# #         img_np_ = img_np_[:slice_width*slice_height//14//14].reshape(slice_height//14,slice_width //14, 14, 14, 3).transpose(0, 2, 1, 3, 4).reshape(slice_height, slice_width, 3) #.transpose(1, 0, 2)
# #     img_ = Image.fromarray(img_np_)
# #     # 变成原来图像

# #     print(img_.size)

# #     img_.save(f"my_{img_name.split('.')[0]}_{i}.png")
# # # print(ind_tokens, len(ind_tokens), len(images))
# # # for i in range(len(images)):
# # #     img_ = images[i]
# # #     to_pil = ToPILImage()
# # #     img_ = to_pil(img_)
# # #     print(img_.size)
# # #     img_.save(f"{img_name.split('.')[0]}_{i}.png")


# source_image, patches, best_grid, ind_tokens = \
#     slice_image_minicpm(image=img, max_slice_nums=6, scale_resolution=336, patch_size=14, never_split=False)
# print(source_image, patches, best_grid, ind_tokens)
# # patches = [item for sublist in patches for item in sublist]

# if len(patches) != 0:
#     xx = image_processor.preprocess(source_image, do_resize=False, 
#                                         do_center_crop=False,
#                                         do_rescale=True,
#                                         do_normalize=True,
#                                         return_tensors='pt')['pixel_values']
#     for i, x in enumerate(xx):
#         to_pil = ToPILImage()
#         img_ = to_pil(x)
#         img_.save(f"mini_post_{img_name.split('.')[0]}_{i}.png")

# for i, p in enumerate(patches):
#     p.save(f"mini_{img_name.split('.')[0]}_{i}.png")


