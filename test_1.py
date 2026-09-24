
# test_1_multidistortion_metric_final_v3.py
# ABINN_Ala/Ours single secret evaluation
# PSNR/SSIM: Y channel
# RMSE/MAE: RGB channel
# Keep original ABINN reverse flow:
# distorted stego image -> DWT -> reverse INN

import os
import math
import cv2
import numpy as np
import torch

from model import *
from distortion import apply_distortion
import config as c
import datasets
import modules.Unet_common as common


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rgb_to_y(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0]


def psnr_y(img1, img2):
    y1 = rgb_to_y(img1).astype(np.float64)
    y2 = rgb_to_y(img2).astype(np.float64)

    mse = np.mean((y1 - y2) ** 2)

    if mse < 1e-12:
        return 100

    return 10 * math.log10(255 ** 2 / mse)


def ssim_y(img1, img2):
    y1 = rgb_to_y(img1).astype(np.float64)
    y2 = rgb_to_y(img2).astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(y1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(y2, (11, 11), 1.5)

    sigma1 = cv2.GaussianBlur(y1 ** 2, (11, 11), 1.5) - mu1 ** 2
    sigma2 = cv2.GaussianBlur(y2 ** 2, (11, 11), 1.5) - mu2 ** 2
    sigma12 = cv2.GaussianBlur(y1 * y2, (11, 11), 1.5) - mu1 * mu2

    return np.mean(
        ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) /
        ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    )


def rmse(img1, img2):
    return np.sqrt(
        np.mean(
            (img1.astype(np.float64) - img2.astype(np.float64)) ** 2
        )
    )


def mae(img1, img2):
    return np.mean(
        np.abs(
            img1.astype(np.float64) - img2.astype(np.float64)
        )
    )


def tensor_to_img(x):
    x = torch.clamp(x, 0, 1)
    x = x.detach().cpu().numpy().transpose(0, 2, 3, 1)
    return (x * 255).astype(np.uint8)


if __name__ == "__main__":

    net = Model_1().cuda()
    init_model(net)

    net = torch.nn.DataParallel(
        net,
        device_ids=c.device_ids
    )

    checkpoint = torch.load(
        c.MODEL_PATH_1 + c.suffix,
        weights_only=False
    )

    state_dict = {
        k: v for k, v in checkpoint["net"].items()
        if "tmp_var" not in k
    }

    net.load_state_dict(state_dict)
    net.eval()

    dwt = common.DWT()
    iwt = common.IWT()

    attacks = [
        "none",
        "gaussian1",
        "gaussian10",
        "poisson",
        "jpeg40",
        "jpeg90"
    ]

    results = {
        attack: {
            "PSNR_Y": [],
            "SSIM_Y": [],
            "RMSE": [],
            "MAE": []
        }
        for attack in attacks
    }

    cover_steg = {
        "PSNR_Y": [],
        "SSIM_Y": [],
        "RMSE": [],
        "MAE": []
    }


    with torch.no_grad():

        for i, data in enumerate(datasets.testloader):

            data = data.cuda()

            cover = data[data.shape[0] // 2:]
            secret = data[:data.shape[0] // 2]

            input_img = torch.cat(
                (
                    dwt(cover),
                    dwt(secret)
                ),
                dim=1
            )

            output, _ = net(input_img)

            output_steg = output.narrow(
                1,
                0,
                4 * c.channels_in
            )

            steg_img = iwt(output_steg)

            cover_np = tensor_to_img(cover)
            steg_np = tensor_to_img(steg_img)

            for b in range(cover_np.shape[0]):

                cover_steg["PSNR_Y"].append(
                    psnr_y(cover_np[b], steg_np[b])
                )

                cover_steg["SSIM_Y"].append(
                    ssim_y(cover_np[b], steg_np[b])
                )

                cover_steg["RMSE"].append(
                    rmse(cover_np[b], steg_np[b])
                )

                cover_steg["MAE"].append(
                    mae(cover_np[b], steg_np[b])
                )


            for attack in attacks:

                # Keep image range valid before applying distortion
                steg_input = torch.clamp(
                    steg_img,
                    0,
                    1
                )

                steg_attack, _ = apply_distortion(
                    steg_input,
                    attack
                )

                # Avoid numerical overflow after distortion
                steg_attack = torch.clamp(
                    steg_attack,
                    0,
                    1
                )

                steg_attack_dwt = dwt(steg_attack)

                # ABINN reverse input channel check
                assert steg_attack_dwt.shape[1] == output_steg.shape[1], \
                    f"DWT channel mismatch: {steg_attack_dwt.shape} vs {output_steg.shape}"

                backward = net(
                    steg_attack_dwt,
                    rev=True
                )

                secret_rev = backward.narrow(
                    1,
                    4 * c.channels_in,
                    backward.shape[1] - 4 * c.channels_in
                )

                secret_rev = iwt(secret_rev)

                secret_np = tensor_to_img(secret)
                recovery_np = tensor_to_img(secret_rev)

                for b in range(secret_np.shape[0]):

                    results[attack]["PSNR_Y"].append(
                        psnr_y(secret_np[b], recovery_np[b])
                    )

                    results[attack]["SSIM_Y"].append(
                        ssim_y(secret_np[b], recovery_np[b])
                    )

                    results[attack]["RMSE"].append(
                        rmse(secret_np[b], recovery_np[b])
                    )

                    results[attack]["MAE"].append(
                        mae(secret_np[b], recovery_np[b])
                    )


    with open(
        "metrics_multidistortion.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("========== Cover-Stego ==========\n")

        for k, v in cover_steg.items():
            f.write(f"{k}: {np.mean(v):.6f}\n")

        for attack in attacks:

            f.write(
                f"\n========== Secret-Recovery-{attack} ==========\n"
            )

            for k, v in results[attack].items():
                f.write(f"{k}: {np.mean(v):.6f}\n")


    print("Finished.")
    print("Saved metrics_multidistortion.txt")
