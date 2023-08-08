"""Drive in multiple games without any hook code.

We will allow ourselves to access only the sights and sounds a human player
would have access to, and the keyboard, mouse, and controller inputs a human
player would be able to make.

To begin with, we'll focus on following a dotted line on the minimap,
which is pretty conserved across games.
"""

import sys
sys.path.append('src')
import gta.recording.vision, matplotlib.pyplot as plt

win = gta.recording.vision.CyberpunkWindow(
    # do_resize_bboxes_interactive=True,
)

fig, (ax, bx) = plt.subplots(ncols=2, figsize=(10, 5))
mask, basemap = win.get_track_mask(
    basemap_kind='minimap',
    do_erode=False,
    filters_present=['cpunk_yellow_dots'],
    do_perspective_transform=False,
)
ax.imshow(basemap, aspect='equal')
bx.imshow(mask, aspect='equal')
plt.show()