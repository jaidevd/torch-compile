from basicsr.models.archs.NAFSSR_arch import NAFSSR
import torch
import cv2
import matplotlib.pyplot as plt


def load_stereo_images(left_path, right_path, device="cpu"):
    x_l = cv2.imread(left_path)
    x_l = cv2.cvtColor(x_l, cv2.COLOR_BGR2RGB)
    x_l = torch.from_numpy(x_l.transpose(2, 0, 1)).float() / 255.0

    x_r = cv2.imread(right_path)
    x_r = cv2.cvtColor(x_r, cv2.COLOR_BGR2RGB)
    x_r = torch.from_numpy(x_r.transpose(2, 0, 1)).float() / 255.0

    x = torch.cat([x_l, x_r], dim=0).unsqueeze(0).to(device)
    return x_l, x_r, x


def load_model(path="experiments/pretrained_models/NAFSSR-L_4x.pth"):
    # This is the original torch.nn.Module
    net_g = NAFSSR(up_scale=4, width=128, num_blks=128)
    checkpoint = torch.load(path)
    net_g.load_state_dict(checkpoint["params"], strict=True)
    return net_g.eval()


if __name__ == "__main__":
    x_l, x_r, x = load_stereo_images("demo/lr_img_l.png", "demo/lr_img_r.png")
    model = load_model()
    # print("Tracing...")
    # traced = torch.jit.trace(model, (x,))
    with torch.no_grad():
        y = model(x)

    y_l = y[0, :3]
    y_r = y[0, 3:]

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(x_l.permute(1, 2, 0))
    ax[0, 0].set_title("Left LR")
    ax[0, 1].imshow(x_r.permute(1, 2, 0))
    ax[0, 1].set_title("Right LR")
    ax[1, 0].imshow(y_l.permute(1, 2, 0))
    ax[1, 0].set_title("Left SR")
    ax[1, 1].imshow(y_r.permute(1, 2, 0))
    ax[1, 1].set_title("Right SR")
    [a.axis("off") for row in ax for a in row]
    plt.tight_layout()
    plt.show()
