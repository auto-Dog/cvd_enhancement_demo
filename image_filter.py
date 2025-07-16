import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch.nn as nn
from colorFilter import colorFilter

pth_optim_location = "model_vit_cn7aE_D100_optim_base.pth"

# 假设 enhancemodel 已经定义
enhancemodel = colorFilter().cuda()
enhancemodel = nn.DataParallel(enhancemodel,device_ids=list(range(torch.cuda.device_count())))
enhancemodel = enhancemodel.cuda()
enhancemodel.load_state_dict(torch.load(pth_optim_location, map_location='cpu'))


def single_enhancement(img: np.ndarray):
    """调用模型，对单张图片进行色彩增强"""
    image_sample = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0)
    image_sample = image_sample.cuda()
    img_out = image_sample.clone()
    # 一次性生成方案：
    enhancemodel.eval()
    with torch.no_grad():
        img_t = enhancemodel(img_out)    # 采用cnn变换改变色彩
    img_out_array = img_t.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img_out_array = np.clip(img_out_array, 0.0, 1.0)
    return img_out_array

def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
         based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))


def get_mapping_func(image1, image2):
    """ Computes the polynomial mapping """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m


def apply_mapping_func(image, m):
    """ Applies the polynomial mapping """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result


def show_color_enhancement(original_image):

    original_size = original_image.size
    # 调整图像大小为 240x240
    small_image = original_image.resize((240, 240))
    small_image_np = np.array(small_image) / 255.0

    # 进行颜色增强
    enhanced_small_image = single_enhancement(small_image_np)

    # 计算映射函数
    mapping = get_mapping_func(small_image_np, enhanced_small_image)

    # 将映射应用到原始大尺寸图像
    original_image_np = np.array(original_image) / 255.0
    enhanced_large_image = apply_mapping_func(original_image_np, mapping)
    enhanced_large_image = np.clip(enhanced_large_image, 0.0, 1.0)
    # 转换成PIL图像
    enhanced_large_image = Image.fromarray(np.uint8(enhanced_large_image * 255))
    enhanced_small_image = Image.fromarray(np.uint8(enhanced_small_image * 255))
    # # 显示原始图像和增强后的图像
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_image_np)
    # plt.title('Original Image')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(enhanced_large_image)
    # plt.title('Enhanced Image')
    # plt.axis('off')

    # plt.show()
    return enhanced_large_image


if __name__ == "__main__":
    # 请替换为你的输入图像路径
    input_image_path = "image_test/flower.png"
        # 读取原始图像
    original_image = Image.open(input_image_path).convert('RGB')
    out_image = show_color_enhancement(original_image)
    out_image.show()


