from natsort import natsorted
import cv2
import os
import utils
# from natsort import natsorted


for ii in range(4, 1112):
    ori_floder = r'/home/hua/Code/xgl/attention_map/ONLformer/featuremaps/attention_map'
    new_floder = r'/home/hua/Code/xgl/attention_map/ONLformer/featuremaps/attention_map_jet'

    # utils.mkdirs(new_floder)
    ori_floder = os.path.join(ori_floder, str(ii))
    new_floder = os.path.join(new_floder, str(ii))

    utils.mkdirs(new_floder)

    file_list = natsorted(os.listdir(ori_floder))
    # print(file_list)
    for file_name in file_list:

        file_path = os.path.join(ori_floder, file_name)
        img_ori = cv2.imread(file_path,2)
        img_jet = cv2.applyColorMap(img_ori, cv2.COLORMAP_JET)
        new_name = file_name.split('.')[0] + '_jet' + '.png'
        # print('new_name', new_name)

        new_path = os.path.join(new_floder, new_name)
        cv2.imwrite(new_path,img_jet)
        print(new_path)
