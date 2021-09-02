import os
import torch
import numpy as np
from PIL import Image


# def convert_rgb_to_y(img):
#     if type(img) == np.ndarray:
#         return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
#     elif type(img) == torch.Tensor:
#         if len(img.shape) == 4:
#             img = img.squeeze(0)
#         return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
#     else:
#         raise Exception('Unknown Type', type(img))


# def convert_rgb_to_ycbcr(img):
#     if type(img) == np.ndarray:
#         y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
#         cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
#         cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
#         return np.array([y, cb, cr]).transpose([1, 2, 0])
#     elif type(img) == torch.Tensor:
#         if len(img.shape) == 4:
#             img = img.squeeze(0)
#         y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
#         cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
#         cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
#         return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
#     else:
#         raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    r = (
        298.082 * img[:, :, 0] / 256.0
        + 408.583 * img[:, :, 2] / 256.0
        - 222.921
    )
    g = (
        298.082 * img[:, :, 0] / 256.0
        - 100.291 * img[:, :, 1] / 256.0
        - 208.120 * img[:, :, 2] / 256.0
        + 135.576
    )
    b = (
        298.082 * img[:, :, 0] / 256.0
        + 516.412 * img[:, :, 1] / 256.0
        - 276.836
    )
    return np.array([r, g, b]).transpose([1, 2, 0])


def calc_psnr(img1, img2):
    return 10.0 * torch.log10(1.0 / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def min_max(m):
    mx = m.max()
    mn = m.min()
    return (m - m.min()) / (mx - mn)


def prepare_images(path_input, path_target, out):
    out = min_max(out)
    input_img = Image.open(path_input)
    target_img = Image.open(path_target)

    target_splits = target_img.convert("YCbCr").split()
    target_y = target_splits[0]

    input_splits = input_img.convert("YCbCr").split()
    input_y = input_splits[0]

    out_image = out.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    # out_image = np.array(
    #     [out_image, np.array(target_splits[1]), np.array(target_splits[2])]
    # ).transpose([1, 2, 0])
    # out_image = convert_ycbcr_to_rgb(out_image)
    out_image = np.clip(out_image, 0.0, 255.0).astype(np.uint8)
    out_image = Image.fromarray(out_image)

    return target_y, input_y, out_image


def save_images(results_dir, path_input, path_target, out, score):

    score = round(score, 3)
    results_dir = os.path.join(results_dir, "images")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    target, input, out_image = prepare_images(path_input, path_target, out)

    target.save(f"{results_dir}/taret_step_{score}.png")
    input.save(f"{results_dir}/input_step_{score}.png")
    out_image.save(f"{results_dir}/out_image_step_{score}.png")


# # Open image
# image = Image.open(args.file).convert("YCbCr")
# image = np.array(image).astype(np.float32)、

# # RGB convert to YCbCr
# y = 16. + (64.738 * image[:, :, 0] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 2]) / 256.
# cb = 128. + (-37.945 * image[:, :, 0] - 74.494 * image[:, :, 1] + 112.439 * image[:, :, 2]) / 256.
# cr = 128. + (112.439 * image[:, :, 0] - 94.154 * image[:, :, 1] - 18.285 * image[:, :, 2]) / 256.
# ycbcr = np.array([y, cb, cr]).transpose([1, 2, 0])

# inputs = ycbcr[..., 0]
# inputs /= 255.
# inputs = torch.from_numpy(inputs).to(device)
# inputs = inputs.unsqueeze(0).unsqueeze(0)

# with torch.no_grad():
#     out = model(inputs).clamp(0.0, 1.0)

# out_image = out.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

# out_image = np.array([out_image, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])

# # YCbCr convert to RGB
# if len(out_image.shape) == 4:
#     out_image = out_image.squeeze(0)
# y = 16. + (64.738 * out_image[0, :, :] + 129.057 * out_image[1, :, :] + 25.064 * out_image[2, :, :]) / 256.
# cb = 128. + (-37.945 * out_image[0, :, :] - 74.494 * out_image[1, :, :] + 112.439 * out_image[2, :, :]) / 256.
# cr = 128. + (112.439 * out_image[0, :, :] - 94.154 * out_image[1, :, :] - 18.285 * out_image[2, :, :]) / 256.
# out_image = torch.cat([y, cb, cr], 0).permute(1, 2, 0)

# out_image = np.clip(out_image, 0.0, 255.0).astype(np.uint8)
# out_image = Image.fromarray(out_image)
# out_img.save(f"srcnn.png")