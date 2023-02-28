# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
from ast import arg
from email.policy import default
import os

import cv2
import numpy as np
import pandas as pd

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.appearence_similarity import Appearence_Similarity
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.appearence_similarity import Appearence_Similarity
import utils
#from deep_sort.utils import rescale_img

def generate_feature_info(frame, bbox):

    pass
def convert_bbox_format(masks):
    output_array = []
    for frame_num, frame in enumerate(masks):
        for i, bbox in enumerate(frame['boxes']):
            y1, x1, y2, x2 = bbox
            w = np.abs(x2-x1)
            h = np.abs(y2-y1)
            #x = (x2-x1)//2+x1
            #y = (y2-y1)//2+y1
            conf = frame['mask_scores'][i]
            output_array.append([frame_num, -1, x1, y1, w, h, conf])
        
        #print(output_array)
    return np.array(output_array)


def gather_sequence_info(sequence_dir, detection_file, video_frames):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.
    video_frames : ndarray 
        Molded video frames

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_filenames = {}

    if sequence_dir:
        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}
    else:
        sequence_dir=''
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file, allow_pickle=True)
        if type(detections[0]) == dict:
            detections = convert_bbox_format(detections)
        print(detections[5])

    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    elif video_frames:
        image_size = len(video_frames[0][0]), len(video_frames[0][1])
    else: 
        image_size=None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms, 
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0, Appearences=None, frame=None):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.
    Appearences :  Optional[Appearence_Similarity]
        Object which can determine features of a frame or bounded box 
    frame : ndarray
        The current frame 
    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int64)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        if not feature.size and Appearences:
            feature = Appearences.get_features(frame, bbox)
            print(len(feature))
        detection_list.append(Detection(bbox, confidence, feature))

    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, output_video, target_file, model_type, generate_tracklets):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.
    output_video : string
        Path to video where the tracked frames will be saved 
    target_file : string
        Path to target file
    model_type : string
        Do appearence similarity, and if so using what model?
    generate_tracklets : bool 
        Create a directory with videos of each tracklet?

    """
    molded_video = None

    if target_file:
        vidcap = cv2.VideoCapture(target_file)
        sub_sample = 1 # sampler
        fps        = int(vidcap.get(cv2.CAP_PROP_FPS)) # We need to get frames per second
        n_frames   = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) # and number of frames

        # Most videos have fps=30 but some have fps=60 (those I downsample)
        if fps == 60:
            sub_sample = 2

        videodata = utils.loadVideo(target_file, greyscale=False)
        videodata = videodata[::sub_sample,...] # do the sampling 
        molded_video = utils.mold_video(videodata, dimension=1280, n_jobs=1)
    else: 
        fps = 30
    seq_info = gather_sequence_info(sequence_dir, detection_file, molded_video)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, n_init=1)
    results = []

    Appearences = Appearence_Similarity(model_type=model_type)

    if output_video:
        out_video = cv2.VideoWriter(output_video, 
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 1280))
        #visualizer.init_save_video(output_video)
    if generate_tracklets:
        tracklet_dir = output_video.split('.')[0]+  "_tracklets"
        if not os.path.exists(tracklet_dir):
            os.mkdir(tracklet_dir)
        tracklet_videos = dict()
    

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        try:
            detections = create_detections(
                seq_info["detections"], frame_idx, min_detection_height, Appearences=Appearences,
                    frame=molded_video[frame_idx])
        except Exception as e: #if no video, I should really clean this up 
            detections = create_detections(
                seq_info["detections"], frame_idx, min_detection_height)
            
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker
        tracker.predict()
        tracker.update(detections)
        # Update visualization.

        if display:
            if target_file and frame_idx < n_frames:
                 image = cv2.cvtColor(molded_video[frame_idx], cv2.COLOR_RGB2BGR)
                #seq_info["image_size"] = image.shape
            else:
                image = cv2.imread(
                    seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            #vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
            image = vis.get_image()
            if output_video:
                out_video.write(image)
                #vis.save_frame()


        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            #bbox_ints = [int(x) for x in bbox]
            bbox_conf = track.current_bbox_confidence
            bbox_confirmed = bbox_conf == 100 
            
            if generate_tracklets:
                #image = cv2.cvtColor(molded_video[frame_idx], cv2.COLOR_RGB2BGR)

                if not tracklet_videos.get(track.track_id):
                    
                    tracklet_videos[track.track_id] = cv2.VideoWriter(
                        os.path.join(tracklet_dir, "track{}.mp4".format(track.track_id)), 
                        cv2.VideoWriter_fourcc(*'mp4v'), fps, (256, 256))
            
                tracklet_image =  (utils.rescale_img(bbox, molded_video[frame_idx])*255).astype(np.uint8)
                
                tracklet_image = cv2.cvtColor(
                   tracklet_image, cv2.COLOR_RGB2BGR)
                
                cv2.imshow('Frame', tracklet_image)
                tracklet_videos[track.track_id].write(tracklet_image)
                

                #tracklet_videos[track.track_id] = current_tracklet

            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], 
                bbox_confirmed, bbox_conf])       

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    try:
        visualizer.run(frame_callback)

    except Exception as e: 
        print('Failed to predict frame')
    if generate_tracklets: 
        for tracklet_vid in tracklet_videos.values():
            tracklet_vid.release()
    if output_video:
        out_video.release()
        #visualizer.end_save_video()
    
    # Store results.
    if output_file:
        output_df = pd.DataFrame(results, 
            columns= ["frame_index", "track_id", "voc_xmin", "voc_ymin", 
                    "voc_xmax", "voc_ymax", "bb_confirmed", "bb_confidence"])

        output_df.to_csv(output_file)



def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None)
    parser.add_argument(
        "--target_file", help="Path to target file",
        default=None)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--output_video", help="Path to the tracking output video",
        default=None)
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--bulk_track", help="Directory where videos and detections are stored",
        default=None)
    parser.add_argument(
        "--model_type", help="Model type for appearence similarity",
        default=None)
    parser.add_argument(
        "--generate_tracklets", help="Create folder containing videos of tracklets",
        default=None, type=bool_string)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.bulk_track:
        videos_dir = os.path.join(args.bulk_track, "videos")
        detections_dir = os.path.join(args.bulk_track, "detections")
        output_dir = os.path.join(args.bulk_track, f"{args.model_type}_outputs")

        video_filenames = {
            os.path.splitext(f)[0].split(".")[0].replace(" ", "_"): os.path.join(videos_dir, f)
            for f in os.listdir(videos_dir)}
        detection_fileneames = {
            os.path.splitext(f)[0].split("_seg")[0].replace(" ", "_"): os.path.join(detections_dir, f)
            for f in os.listdir(detections_dir)}
        for key in video_filenames:
            output_video = os.path.join(output_dir, f'{key}_tracked_{args.model_type}.mp4')
            output_file = os.path.join(output_dir, f'{key}_tracked_{args.model_type}.csv')
            if os.path.exists(detection_fileneames.get(key, "none")) and os.path.exists(video_filenames[key]) and not os.path.exists(output_video) :
                #only run if both detection file and video file exist 
                print(f"file root: {key}, video_path : {video_filenames[key]}, detection path : {detection_fileneames[key]}")
                
                print(output_video)
                run(
                    args.sequence_dir, detection_fileneames[key], output_file,
                    args.min_confidence, args.nms_max_overlap, args.min_detection_height,
                    args.max_cosine_distance, args.nn_budget, args.display, output_video, 
                    video_filenames[key], args.model_type, args.generate_tracklets)
    else:
        run(
            args.sequence_dir, args.detection_file, args.output_file,
            args.min_confidence, args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, args.display, args.output_video, 
            args.target_file, args.model_type, args.generate_tracklets)
