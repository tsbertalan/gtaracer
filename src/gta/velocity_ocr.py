import numpy as np

# try:
from PIL import Image
# except ImportError:
# import Image
import pytesseract

from tqdm.auto import tqdm
from PIL import ImageEnhance
import PIL.Image
from pytesseract import image_to_string

from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

from os.path import join, expanduser, dirname
HERE = dirname(__file__)
from sys import path
path.append(join(HERE, '..'))

import gta.robust_spline

def load_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    Y = data['Y']
    X1, X2 = X.T
    
    image_times = X1.astype('float')

    dts = np.diff(image_times)[:10]
    np.mean(dts), np.std(dts)
    # X2.shape, X2[:4]
    images = np.stack(X2).astype('uint8')

    Y1, Y2 = Y

    axis_times = np.array([t for (t, (aid, aval)) in Y1])
    axis_ids = np.array([aid for (t, (aid, aval)) in Y1])
    axis_vals = np.array([aval for (t, (aid, aval)) in Y1])

    return {
        'image_times': image_times, 'images': images, 
        'axis_times': axis_times, 'axis_ids': axis_ids, 'axis_vals': axis_vals,
        'button_data': Y2,
    }


def get_miles_subimage(im):
    subarr = im[1085:1115, 110:180]#.astype('float')
    return subarr#.astype('uint8')


def mode(a, bins=32):
    counts, bins = np.histogram(a.ravel(), bins=bins)
    centers = bins[:-1] + np.diff(bins) / 2.
    return centers[np.argmax(counts)]
    

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def boost_contrast(img, factor):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def sharpen(img, factor):
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)


def sanitize(s, a=True, A=True, n=True, extra=' ', replace=''):
    allowed = extra
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    if a:
        allowed += alpha
    if A:
        allowed += alpha.upper()
    if n:
        allowed += '1234567890'
    return ''.join([
        (c if c in allowed else replace)
        for c in s
    ])


def enhance(img, upscale=4, grey=True, contrast_factor=1.2, 
            sharpness_factor=1, otsu=False, array=True, 
            binarize_level=170, flip=True):
    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img)

    # Do operations on PIL Image.
    if upscale > 1:
        new_size = tuple(upscale*x for x in img.size)
        img = img.resize(new_size, PIL.Image.ANTIALIAS)
    if contrast_factor != 1:
        img = boost_contrast(img, contrast_factor)
    if sharpness_factor != 1:
        img = sharpen(img, sharpness_factor)
    if grey:
        img = img.convert('L')

    # Do operations on ndarray.
    if otsu:
        from skimage import filters
        img = np.array(img)
        val = filters.threshold_otsu(img)
        img = (img < val).astype('uint8') * 255

    img = np.asarray(img)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], -1)

    if binarize_level:
        img[img < binarize_level] = 0
        img[img >= binarize_level] = 255
        
    if flip:
        img = (255 - img.astype(int)).astype('uint8')
        
    # Maybe return PIL Image.
    if array:
        return img
    else:
        return PIL.Image.fromarray(img)


def get_text(img, 
    ascii=True, 
    dictionary=False, whitelist=None, OCR_engine_mode=1, page_segmentation_mode=12,
    std_threshold=7,
    **enhance_kw
    ):
    enhanced_img = enhance(img, **enhance_kw)

    # Short-circuit if the image is very smooth.
    stdkw = {}
    stdkw.update(enhance_kw)
    stdkw['sharpness_factor'] = 0
    std = enhance(img, **stdkw, array=True).std()
    if std < std_threshold:
        return '', enhanced_img

    # Generate Tesseract config options
    # see https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
    config = '--oem %d --psm %d' % (OCR_engine_mode, page_segmentation_mode)
    if not dictionary:
        config = '%s -c load_system_dawg=0 -c load_freq_dawg=0' % config
    if whitelist is not None:
        config = '%s -c tessedit_char_whitelist=%s' % (config, whitelist)
    
    out = image_to_string(enhanced_img, config=config)

    if ascii:
        out = out.encode('ascii', errors='ignore').decode('ascii')

    return out, enhanced_img


def get_dist(im):
    t, im = get_text(get_miles_subimage(im))
    ts = sanitize(t, extra='.')
    n = sanitize(t, a=False, A=False, extra='.')
    
    # Ignore periods after the first one.
    period_tokens = n.split('.')
    n = '.'.join(period_tokens[:2])
    if len(period_tokens) > 2:
        n = n + ''.join(period_tokens[2:])
        
    unit = 'mi' if 'm' in ts else ('ft' if 'f' in ts else '?')
    
    n = '' if n == '.' else n  # Give up if we just see a period.
    
    d = None if not n else float(n)
    return d, unit


def process_npz(data_path, max_images=9999999, skip=1):
    data = load_data(data_path)

    images = data['images'][:max_images][::skip]
    image_times = data['image_times'][:max_images][::skip]
    mean_dt = np.mean(np.diff(image_times))
    std_dt = np.std(np.diff(image_times))
    print('Time between images with this skip factor is %.2f+/-%.2f seconds' % (mean_dt, std_dt))

    dists_units = []
    for im in tqdm(images, unit='images', desc='Extracting distances'):
        dists_units.append(get_dist(im))

    dists_any_units = np.array([d for (d,u) in dists_units]).astype(float)
    dists_any_units[dists_any_units > 10.0] = np.nan

    units = [u for (d,u) in dists_units]

    dists_miles = np.array([
        d if u == 'mi' else float(d) / 5280.
        for (d,u) in zip(dists_any_units, units)
    ])

    ok = np.argwhere(np.logical_not(np.isnan(dists_miles))).ravel()
    # cubic = PolynomialFeatures(degree=3)
    # cubic_features = cubic.fit_transform(image_times[ok].reshape((-1, 1)))

    # try:
    #     ransac = RANSACRegressor(LinearRegression(), random_state=0) 
    #                         #  max_trials=100, 
    # #                          min_samples=50, 
    # #                          residual_metric=lambda x: np.sum(np.abs(x), axis=1), 
    # #                          residual_threshold=5.0, 
    #                         # random_state=0)
    #     ransac.fit(cubic_features, dists_miles[ok].reshape((-1, 1)))
    # except ValueError:
    #     # RANSAC failed; loosen the threshold.
    #     ransac = RANSACRegressor(LinearRegression(), random_state=0, residual_threshold=.1)
    # inliers = np.argwhere(ransac.inlier_mask_).ravel()

    # Use a particular knot spacing in seconds.
    recording_length = image_times.max() - image_times.min()
    target_knots_1 = int(recording_length / 5.38)
    # target_knots_2 = int(recording_length / 12.1)

    # Fit one "robust spline" just so we know which are the inliers.
    rspline = gta.robust_spline.RobustSpline(residual_threshold=.01, n_knots=target_knots_1)
    rspline.fit(image_times[ok], dists_miles[ok])
    inliers = np.argwhere(rspline.inliers).ravel()

    # Use just these inliers to fit our real spline(s), with alternate parameters.
    t_cleaned = image_times[ok][inliers]
    tmin = t_cleaned.min()
    t_cleaned -= tmin
    d_cleaned = dists_miles[ok][inliers]
    t_all = image_times - tmin

    # # Fit a second rspline with a different knot spacing.
    # rspline_2 = gta.robust_spline.RobustSpline(residual_threshold=.01, n_knots=target_knots_2)
    # rspline_2.fit(t_cleaned, d_cleaned)

    # Fit a vanilla sklearn spline with a more reasonable number of knots.
    # (Also, I don't know how to get the derivative of the rspline.)
    spline_smoothing_factor = .01  # This is a minimum error bound for the spline fitting the data,
    # and the number of knots will adjust to meet it. If it's zero, then the spline will go through every point.
    spl = UnivariateSpline(t_cleaned, d_cleaned, s=spline_smoothing_factor)

    velocities = spl.derivative(1)(t_all)
    v_cleaned = spl.derivative(1)(t_cleaned)
    d_fit = spl(t_cleaned)
    d_all = spl(t_all)

    # fig, ax = plt.subplots()
    # ax.scatter(image_times[ok][inliers].ravel(), dists_miles[ok][inliers].ravel(), marker='o', label='inliers')
    # ax.plot(image_times.ravel(), rspline.predict(image_times.reshape((-1, 1))).ravel(), linewidth=5, alpha=.6, label='Inlier-Detection RSpline')
    # ax.plot(t_all+tmin, rspline_2.predict(t_all.reshape((-1, 1))).ravel(), linewidth=5, alpha=.6, label='RSpline with fewer knots')
    # ax.plot(t_all+tmin, spl(t_all.reshape((-1, 1))).ravel(), linewidth=5, alpha=.6, linestyle='--', label='USpline, $s=%f$' % spline_smoothing_factor)
    # ax.legend()
    # ax.set_ylim(.7, 1.3)
    # fig.savefig('recordings/check_data.png')

    return {
        'images': images,

        't_cleaned': t_cleaned,
        'd_cleaned': d_cleaned,
        'v_cleaned': v_cleaned,
        'd_fit': d_fit,

        't': t_all,
        'd': d_all,
        'v': velocities,
    }


def reduce_npz(data_path, img_height=32, **kw_process):
    assert data_path.endswith('.npz')

    processed = process_npz(data_path, **kw_process)

    n, h, w, c = processed['images'].shape
    aspect_ratio = w / h
    img_width = int(img_height * aspect_ratio)

    import skimage.transform
    scaled_images = np.stack([
        skimage.transform.resize(img, (img_height, img_width))
        for img in tqdm(processed['images'], desc='Rescaling images', unit='image')
    ])

    to_save = {
        k: v
        for k, v in processed.items()
        if k not in ['images',]
    }
    to_save['images'] = scaled_images

    save_path = data_path.replace('.npz', '.reduced.npz')
    np.savez_compressed(save_path, **to_save)

    print('Reduced data saved to %s' % save_path)
    return save_path, processed


def main(figdir, data_path):


    from gta.utils import mkdir_p
    mkdir_p(figdir)

    # 2021-08-22-13-01-57-gtav_recording.npz                              2021-08-22-14-13-41-gtav_recording.npz                              2021-08-22-14-34-09-gtav_recording.npz

    save_path, processed = reduce_npz(data_path, skip=1)

    t = processed['t']
    velocities = processed['v']

    t_cleaned = processed['t_cleaned']
    d_cleaned = processed['d_cleaned']
    d_fit = processed['d_fit']
    v_cleaned = processed['v_cleaned']

    fig, ax = plt.subplots()
    ax.plot(t, velocities)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [mi/hr]')
    fig.tight_layout()
    fig.savefig(join(figdir, 'velocity.png'))

    fig, (ax, bx) = plt.subplots(ncols=2)
    ax.scatter(t_cleaned, d_cleaned, s=1)
    ax.plot(t_cleaned, d_fit, linewidth=4, alpha=.5, color='red')
    ax.set_ylabel('Distance to destination [mi]')

    bx.plot(t_cleaned, v_cleaned * 3600.)
    bx.set_ylabel('Speed [mi/hr]')

    for a in ax, bx:
        a.set_xlabel('Time [s]')

    fig.tight_layout()
    fig.savefig(join(figdir, 'vel_cleaned.png'))
    

if __name__ == '__main__':
    
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--figdir', type=str, default=join(HERE, '..', '..', 'doc', 'figures'))

    parser.add_argument('--data_path', type=str, default=join(
        expanduser('~'), 'data', 'gta', 'velocity_prediction',
    ))
    args = parser.parse_args()
    figdir = args.figdir
    data_path = args.data_path

    if data_path.endswith('.npz'):
        data_path, processed = reduce_npz(data_path)

    else:
        from glob import glob
        search_path = join(data_path, '*-gtav_recording.npz')
        data_paths = glob(search_path)
        print('Found', len(data_paths), 'data files in', search_path, '.')
        for data_path_single in tqdm(data_paths, desc='Processing data', unit='file'):
            main(figdir, data_path_single)


