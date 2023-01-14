import torch

def AngleAxisRotatePoint(angleAxis, pt):
    """
    Use angle_axis vector rotate 3D points
    Parameters:
    ------------
    angleAxis : torch.tensor L*3
    pt : torch.tensor L*3
    """
    theta2 = (angleAxis * angleAxis).sum(dim=1)

    mask = (theta2 > 0).float()

    theta = torch.sqrt(theta2 + (1 - mask))

    mask = mask.reshape((mask.shape[0], 1))
    mask = torch.cat([mask, mask, mask], dim=1)

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    thetaInverse = 1.0 / theta

    w0 = angleAxis[:, 0] * thetaInverse
    w1 = angleAxis[:, 1] * thetaInverse
    w2 = angleAxis[:, 2] * thetaInverse

    wCrossPt0 = w1 * pt[:, 2] - w2 * pt[:, 1]
    wCrossPt1 = w2 * pt[:, 0] - w0 * pt[:, 2]
    wCrossPt2 = w0 * pt[:, 1] - w1 * pt[:, 0]

    tmp = (w0 * pt[:, 0] + w1 * pt[:, 1] + w2 * pt[:, 2]) * (1.0 - costheta)

    r0 = pt[:, 0] * costheta + wCrossPt0 * sintheta + w0 * tmp
    r1 = pt[:, 1] * costheta + wCrossPt1 * sintheta + w1 * tmp
    r2 = pt[:, 2] * costheta + wCrossPt2 * sintheta + w2 * tmp

    r0 = r0.reshape((r0.shape[0], 1))
    r1 = r1.reshape((r1.shape[0], 1))
    r2 = r2.reshape((r2.shape[0], 1))

    res1 = torch.cat([r0, r1, r2], dim=1)

    wCrossPt0 = angleAxis[:, 1] * pt[:, 2] - angleAxis[:, 2] * pt[:, 1]
    wCrossPt1 = angleAxis[:, 2] * pt[:, 0] - angleAxis[:, 0] * pt[:, 2]
    wCrossPt2 = angleAxis[:, 0] * pt[:, 1] - angleAxis[:, 1] * pt[:, 0]

    r00 = pt[:, 0] + wCrossPt0
    r01 = pt[:, 1] + wCrossPt1
    r02 = pt[:, 2] + wCrossPt2

    r00 = r00.reshape((r00.shape[0], 1))
    r01 = r01.reshape((r01.shape[0], 1))
    r02 = r02.reshape((r02.shape[0], 1))

    res2 = torch.cat([r00, r01, r02], dim=1)

    return res1 * mask + res2 * (1 - mask)


def coord_normalization(keypoints, h, w, scale=1):
    """
    Normalize keypoints to [-1, 1] for different scales.
    Parameters:
    ---------------
    keypoints: torch.tensor N*2
        coordinates at different images scales
    """
    keypoints = keypoints - scale / 2 + 0.5  # calc down-sampled keypoints positions
    rescale_tensor = torch.tensor([(w - 1) * scale, (h - 1) * scale]).to(keypoints)
    if len(keypoints.shape) == 2:
        rescale_tensor = rescale_tensor[None]
    elif len(keypoints.shape) == 4:
        # grid scenario
        rescale_tensor = rescale_tensor[None, None, None]
    else:
        raise NotImplementedError
    keypoints /= rescale_tensor
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    return keypoints
