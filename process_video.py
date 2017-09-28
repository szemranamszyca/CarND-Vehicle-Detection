from sklearn.externals import joblib
from feature_extract import *
from find_cars import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from sliding_window import draw_boxes
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


ystart_ystop_scale = [(350, 550, 1.5), (400, 620, 2), (440, 700, 2.5)]


class HeatingControl:

    def __init__(self):
        self.ticks_counter = 1
        self.all_predicted_boxes = []
        self.heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    def tick(self):
        self.ticks_counter += 1

    def reset(self):
        self.ticks_counter = 1
        self.heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        del self.all_predicted_boxes[:]

    def addBoxes(self, boxes):
        self.all_predicted_boxes.extend(boxes)

heating_controler = HeatingControl()

def process_img(img):

    for (ystart, ystop, scale) in ystart_ystop_scale:

        # all_predicted_boxes, all_boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
        #                     hist_bins, all_box=True)
        boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                            hist_bins)
        heating_controler.addBoxes(boxes)

    # print("Counter: ", heating_controler.ticks_counter)
    # print("Boxes", heating_controler.all_predicted_boxes)
    heat = add_heat(heating_controler.heat, heating_controler.all_predicted_boxes)
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    heating_controler.tick()

    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(draw_img)
    # plt.title('Car Positions')
    # plt.subplot(122)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    # fig.tight_layout()
    # plt.show()

    if heating_controler.ticks_counter == 10:
        heating_controler.reset()

    return draw_img

video_output1 = 'project_video_output.mp4'
video_input1 = VideoFileClip('project_video.mp4')#.subclip(40,45)
processed_video = video_input1.fl_image(process_img)
processed_video.write_videofile(video_output1, audio=False)

#
# result = process_img(img)
# plt.imshow(result)
# plt.show()