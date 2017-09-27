from sklearn.externals import joblib
from feature_extract import *
from find_cars import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

color_space='YCrCb'
spatial_size=(32, 32)
hist_bins=32
orient=9
pix_per_cell=8
cell_per_block=2
file_classifier_name = 'svm_class.pkl'
file_scaler_name = 'svm_scaler.pkl'

svc = joblib.load(file_classifier_name)
X_scaler = joblib.load(file_scaler_name)
img = mpimg.imread('./test_images/test6.jpg')

ystart = 400
ystop = 656
scale = 1.5



def process_img(img):
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)
    heat = add_heat(heat, box_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img

video_output1 = 'project_video_output.mp4'
video_input1 = VideoFileClip('project_video.mp4')#.subclip(5,15)
processed_video = video_input1.fl_image(process_img)
processed_video.write_videofile(video_output1, audio=False)


#
# result = process_img(img)
# plt.imshow(result)
# plt.show()