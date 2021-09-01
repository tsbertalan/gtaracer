import numpy as np, cv2, time
from os.path import dirname, join
HERE = dirname(__file__)

from sys import path
path.append(join(HERE, ".."))

import gta.recording.vision


if __name__ == "__main__":
    gameWindow = gta.recording.vision.GtaWindow()

    # tm, basemap = gameWindow.get_track_mask(basemap_kind='minimap', do_erode=False)
    # tm_color = np.tile(tm[..., np.newaxis], (1, 1, 3)).astype(np.uint8) * 255
    # print(tm_color.shape, tm_color.dtype)
    # print(basemap.shape, basemap.dtype)

    while True:
        try:
            mm = np.array(gameWindow.minimap)

            mm = gameWindow.minimap_perspective_transform(mm, visualize=True)
            time.sleep(0.3)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.imshow(mm)
    # ax.grid(True)
    # plt.show()
    
    


    # print(tm_color.shape, basemap.shape)

    # stacked = np.hstack((basemap, tm_color))
    # cv2.imshow('stacked', stacked, stacked.shape[:2])
    # cv2.waitKey(0)


    # cv2.imshow('img', mm)
    # cv2.waitKey(0)

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break

    del gameWindow
