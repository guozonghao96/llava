# import cv2
# import numpy as np 

# my = np.load('myattn.npy')
# now = np.load('nowattn.npy')
# causal = np.load('causal_attn.npy')

# print(my.shape)
# print(now.shape)

# import matplotlib.pyplot as plt


# index = -6

# plt.figure('2')
# plt.imshow(my[index])
# plt.savefig('my')

# plt.figure('1')
# plt.imshow(now[index][0])
# plt.savefig('now')


# plt.figure('3')
# plt.imshow(causal[index][0])
# plt.savefig('causal')



# import numpy as np

# import matplotlib.pyplot as plt

# attn = np.load('attn.npy')
# print(attn.shape)
# plt.figure(figsize=(20, 20))
# for i, att in enumerate(attn):
#     plt.subplot(4, 2, i + 1)
#     plt.imshow(att)
#     print(att)
# plt.savefig('mult_attn.png')




import json
annos = json.load(open('/home/guozonghao/LLaVA-UHD/playground/data/llava_v1_5_mix665k.json'))

new_annos = [

]
for ann in annos:
    if 'image' in ann.keys():
        if 'textvqa' in ann['image']:
            new_annos.append(ann)
            break

with open('textvqa.json', 'w') as f:
    json.dump(new_annos, f)