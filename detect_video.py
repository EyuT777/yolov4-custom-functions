#To run on command line
#python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi --count --detect_item person
import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time, requests
import pandas as pd
from datetime import datetime
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes 
from core.functions import (vehicle_exp_csv, 
                            continious_person_csv, 
                            count_objects, 
                            crop_objects,
                            daily_human_csv,
                            daily_timed, 
                            get_location, 
                            get_current_weather,
                            )
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import time
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video, IP of IP camera or set to 0 for webcam')
flags.DEFINE_string('output', '.', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')
flags.DEFINE_string('detect_item', 'person', 'person or vehicle')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture('C:\\Users\\Eyobed T. Kebede\\Desktop\\yolov4_custom\\archive\\Transport problem in Addis Ababa.mp4')
        #To caputre from IP camera use the input https://username:password@ipaddress:port/video
        #vid = cv2.VideoCapture('https://ipcam1:ipcam1@192.168.0.101:8080/video')
    except:
        vid = cv2.VideoCapture(int(0))

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    data_start_time = time.time()
    while True:
       
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        if FLAGS.detect_item == 'person':
            allowed_classes = ['person']
        elif FLAGS.detect_item == 'vehicle':
            allowed_classes= ['car', 'truck', 'bus', 'motorcycle']

        # if crop flag is enabled, crop each detection and save it as new image
        if FLAGS.crop:
            crop_rate = 150 # capture images every so many frames (ex. crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass          
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
            else:
                pass
        # count objects found
        if FLAGS.count:
            #instantiate the functions defined.
            try:
                current_loc = get_location()
                print('location found: {}\n'.format(current_loc))
                current_weather = get_current_weather(loc_function = current_loc, app_id = '88302dc2ef3a75d9eeda690d201a8cb6')
                print('weather fetched: {}\n'.format(current_weather))
            except Exception as e:
                current_loc = 'None'
                current_weather = 'None'
                print(strftime(e))
            
        
            counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
            now_time = time.time()
            for key, value in counted_classes.items():
                print("Number of {}s: {} ".format(key, value))
            
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        else:
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
    
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            
        if now_time - data_start_time >= 10: #Data per 10 Seconds
            print(daily_timed)
            # call the function to export
            if FLAGS.detect_item.lower() == 'person':
                print('Counting People at time interval {}' .format(now_time-data_start_time))
                continious_person_csv(dicts = counted_classes, weather_data = current_weather, loc_data = current_loc)
                data_start_time = now_time
                    
            elif FLAGS.detect_item.lower() == 'vehicle':
                print('Counting Vehicles time interval {}' .format(now_time-data_start_time))
                vehicle_exp_csv(dicts = counted_classes, weather_data = current_weather, loc_data = current_loc)  
                data_start_time = now_time
            
            #This is for the daily csv summarized export            
        if datetime.now().strftime('%X') == '23:59:59' and FLAGS.detect_item.lower() == 'person':
            daily_human_csv(daily_dict = daily_timed, weather_data = current_weather, loc_data = current_loc)

    
                        
                    
                    
            # loop through dict and print
        	
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
