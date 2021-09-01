import matplotlib as mlp
import matplotlib.pyplot as plt
import random
import numpy as np
import copy

def grid_image(np_images, gts, preds, n=16, shuffle=False):
    """
    np_images 에 저장된 이미지 중 n 개를 뽑아서 figure 에 그린다.
    (정답과 예측 레이블을 표기해준다.)
    
    Args:
        np_images
        gts
        preds
        n
        shuffle
    """
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.sample(range(batch_size), k=n)
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T

    n_grid = np.ceil(n ** 0.5)
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        title = f"gt: {gt}, pred: {pred}"

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def denormalize_image(image, mean, std):
    img_cp = image.copy()
    img_cp *= std
    img_cp += mean
    img_cp *= 255.0
    img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
    # print(np.max(img_cp), np.min(img_cp))
    return img_cp