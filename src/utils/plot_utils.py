import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import random
import cv2
import matplotlib.cm as cm
import math
from einops import repeat
from PIL import Image
import h5py
import os.path as osp
from pathlib import Path

matplotlib.use("Agg")

jet = cm.get_cmap("jet")  # "Reds"
jet_colors = jet(np.arange(256))[:, :3]  # color list: normalized to [0,1]


def blend_img_heatmap(img, heatmap, alpha=0.5):  # (H, W, *)  # (H, W, 3)
    h, w = heatmap.shape[:2]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 2:
        img = repeat(img, "h w -> h w c", c=3)
    img = np.uint8(img)
    heatmap = np.uint8(heatmap * 255)
    blended = np.asarray(
        Image.blend(Image.fromarray(img), Image.fromarray(heatmap), alpha)
    )
    return blended


def error_colormap(x, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    return np.clip(
        np.stack([2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x) * alpha], -1),
        0,
        1,
    )


def plot_image_pair(imgs, dpi=100, size=6, pad=0.5, horizontal=False):
    n = len(imgs)
    assert n == 2, "number of images must be two"
    if horizontal:
        figsize = (size * n, size) if size is not None else None
        _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    else:
        figsize = (size, size * n) if size is not None else None
        _, ax = plt.subplots(n, 1, figsize=figsize, dpi=dpi)

    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap("gray"), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_single_image(img, dpi=100, size=6, pad=0.5):
    figsize = (size, size) if size is not None else None
    _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    ax.imshow(img)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    for spine in ax.spines.values():  # remove frame
        spine.set_visible(False)
    plt.tight_layout(pad=pad)
    fig = plt.gcf()
    return fig


def plot_keypoints(kpts0, kpts1, color="w", ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps, marker="x")
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps, marker="x")


def plot_keypoints_for_img0(kpts, color="w", ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts[:, 0], kpts[:, 1], c=color, s=ps)


def plot_keypoints_for_img1(kpts, color="w", ps=2):
    ax = plt.gcf().axes
    ax[1].scatter(kpts[:, 0], kpts[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=0.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [
        matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]),
            (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1,
            transform=fig.transFigure,
            c=color[i],
            linewidth=lw,
        )
        for i in range(len(kpts0))
    ]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_local_windows(kpts, color="r", lw=1, ax_=0, window_size=9):
    ax = plt.gcf().axes

    patches = []
    for kpt in kpts:
        # patches.append(matplotlib.patches.Rectangle((kpt[0],kpt[1]),window_size,window_size))
        ax[ax_].add_patch(
            matplotlib.patches.Rectangle(
                (kpt[0] - (window_size // 2) - 1, kpt[1] - (window_size // 2) - 1),
                window_size + 2,
                window_size + 2,
                lw=lw,
                color=color,
                fill=False,
            )
        )
    # ax[ax_].add_collection(matplotlib.collections.PathCollection(patches))


def make_matching_plot(
    image0,
    image1,
    kpts0,
    kpts1,
    mkpts0,
    mkpts1,
    color,
    text,
    path=None,
    show_keypoints=False,
    fast_viz=False,
    opencv_display=False,
    opencv_title="matches",
    small_text=[],
):

    if fast_viz:
        make_matching_plot_fast(
            image0,
            image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            path,
            show_keypoints,
            10,
            opencv_display,
            opencv_title,
            small_text,
        )
        return

    plot_image_pair([image0, image1])  # will create a new figure
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color="k", ps=4)
        plot_keypoints(kpts0, kpts1, color="w", ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = "k" if image0[:100, :150].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )

    txt_color = "k" if image0[-100:, :150].mean() > 200 else "w"
    fig.text(
        0.01,
        0.01,
        "\n".join(small_text),
        transform=fig.axes[0].transAxes,
        fontsize=5,
        va="bottom",
        ha="left",
        color=txt_color,
    )
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        # TODO: Would it leads to any issue without current figure opened?
        return fig

def make_matching_plot_fast(
    image0,
    image1,
    kpts0,
    kpts1,
    mkpts0,
    mkpts1,
    color,
    text,
    path=None,
    show_keypoints=False,
    margin=10,
    opencv_display=False,
    opencv_title="",
    small_text=[],
):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin :] = image1
    out = np.stack([out] * 3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1, lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(
            out,
            (x0, y0),
            (x1 + margin + W0, y1),
            color=c,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640.0, 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(
            out,
            t,
            (int(8 * sc), Ht * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * sc,
            txt_color_bg,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (int(8 * sc), Ht * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0 * sc,
            txt_color_fg,
            1,
            cv2.LINE_AA,
        )

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(
            out,
            t,
            (int(8 * sc), int(H - Ht * (i + 0.6))),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5 * sc,
            txt_color_bg,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (int(8 * sc), int(H - Ht * (i + 0.6))),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5 * sc,
            txt_color_fg,
            1,
            cv2.LINE_AA,
        )

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def reproj(K, pose, pts_3d):
    """ 
    Reproj 3d points to 2d points 
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3]
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K

    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose

    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    depth = reproj_points[2]  # [N]
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T

    return reproj_points, depth  # [n, 2]


@torch.no_grad()
def draw_reprojection_pair(
    data, visual_color_type="conf"
):
    figures = {"evaluation": []}

    m_bids = data["m_bids"].cpu().numpy()
    query_image = (data["query_image"].cpu().numpy() * 255).round().astype(np.int32)
    mkpts_3d = data["mkpts_3d_db"].cpu().numpy()
    mkpts_query_c = data["mkpts_query_c"].cpu().numpy()
    mkpts_query = data["mkpts_query_f"].cpu().numpy()
    query_K = data["query_intrinsic"].cpu().numpy()
    query_pose_gt = data["query_pose_gt"].cpu().numpy()  # B*4*4
    m_conf = data["mconf"].cpu().numpy()

    R_errs = data["R_errs"] if "R_errs" in data else None
    t_errs = data["t_errs"] if "t_errs" in data else None
    inliers = data["inliers"] if "inliers" in data else None

    for bs in range(data["query_image"].size(0)):
        mask = m_bids == bs

        mkpts3d_reprojed, depth = reproj(query_K[bs], query_pose_gt[bs], mkpts_3d[mask])
        mkpts_query_masked = mkpts_query[mask]

        if "query_image_scale" in data:
            mkpts3d_reprojed = (
                mkpts3d_reprojed / data["query_image_scale"][bs].cpu().numpy()[[1, 0]]
            )
            mkpts_query_masked = (
                mkpts_query_masked / data["query_image_scale"][bs].cpu().numpy()[[1, 0]]
            )

        text = [
            f"Num of matches: {mkpts3d_reprojed.shape[0]}",
        ]

        if R_errs is not None:
            text += [f"R_err: {R_errs[bs]}"]
        if t_errs is not None:
            text += [f"t_err: {t_errs[bs]}"]
        if inliers is not None:
            text += [
                f"Num of inliers: {inliers[bs].shape[0] if not isinstance(inliers[bs], list) else len(inliers[bs])}"
            ]

        # Clip reprojected keypoints
        mkpts3d_reprojed[:, 0] = np.clip(
            mkpts3d_reprojed[:, 0], a_min=0, a_max=data["query_image"].shape[-1] - 1
        )  # x
        mkpts3d_reprojed[:, 1] = np.clip(
            mkpts3d_reprojed[:, 1], a_min=0, a_max=data["query_image"].shape[-2] - 1
        )  # y

        if visual_color_type == "conf":
            if mkpts3d_reprojed.shape[0] != 0:
                m_conf_max = np.max(m_conf[mask])
                m_conf_min = np.min(m_conf[mask])
                m_conf_normalized = (m_conf[mask] - m_conf_min) / (
                    m_conf_max - m_conf_min + 1e-4
                )
                color = jet(m_conf_normalized)

                text += [
                    f"Max conf: {m_conf_max}",
                    f"Min conf: {m_conf_min}",
                ]
            else:
                color = np.array([])
        elif visual_color_type == "distance_error":
            color_thr = 5
            reprojection_dictance = np.linalg.norm(
                mkpts3d_reprojed - mkpts_query_masked, axis=-1
            )
            color = np.clip(reprojection_dictance / (color_thr), 0, 1)
            color = error_colormap(1 - color, alpha=0.5)
        elif visual_color_type == "depth":
            if depth.shape[0] != 0:
                depth_max = np.max(depth)
                depth_min = np.min(depth)
                depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-4)
                color = jet(depth_normalized)
            else:
                color = np.array([])

        else:
            raise NotImplementedError

        figure = make_matching_plot(
            query_image[bs][0],
            query_image[bs][0],
            mkpts_query_masked,
            mkpts3d_reprojed,
            mkpts_query_masked,
            mkpts3d_reprojed,
            color=color,
            text=text,
        )
        figures["evaluation"].append(figure)

        return figures

def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))