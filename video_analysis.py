import os
import mediapipe as mp
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

left_hand = [18, 20, 16, 22]
right_hand = [15, 21, 17, 19]
face = [i for i in range(11)]
torso = [12, 11, 23, 24]

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def get_poses(video_file_path, show=False, output_file_path="output.mp4"):
    model_path = os.getenv("VISION_MODEL", "pose_landmarker.task")
    
    # Load an mp4 file instead of capturing from the webcam
    base_options = mp.tasks.BaseOptions
    pose_landmarker = mp.tasks.vision.PoseLandmarker
    pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
    vision_running_mode = mp.tasks.vision.RunningMode

    options = pose_landmarker_options(
        base_options=base_options(model_asset_path=model_path),
        running_mode=vision_running_mode.VIDEO)
    
    cap = cv2.VideoCapture(video_file_path)

    # Get the original video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Fixed property name
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Fixed property name
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        fps = 30  # Set a default FPS if reading fails

    # Define video writer to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_file_path, 
        fourcc, 
        fps, 
        (frame_width, frame_height)  # Fixed order: width then height
    )
    
    if not out.isOpened():
        raise RuntimeError("Failed to open output video file")

    pose_results = []
    with pose_landmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read() 
            if not ret:
                break
            
            if frame is None:
                continue

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_idx)
            pose_results.append(pose_landmarker_result)

            if pose_landmarker_result.pose_landmarks:
                # Draw on RGB frame
                drawn_image = draw_landmarks_on_image(rgb_frame, pose_landmarker_result)
                # Convert back to BGR for OpenCV
                drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)
            else:
                drawn_image = frame  # No landmarks detected, keep original frame

            # Write frame to output video
            out.write(drawn_image)

            # Show video with pose overlay (optional)
            if show:
                cv2.imshow("Pose Detection", drawn_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
    return pose_results

def x_y_from_poses(pose_results) -> list[list[tuple]]: 
    ret= []
    threshold = 0.3
    for pose in pose_results:
        if pose.pose_world_landmarks:
            v =[((l.x,l.y) if l.visibility > threshold and l.presence > threshold else None) for l in pose.pose_world_landmarks[0]]
        else:
            v = [None for _ in range(33)]
        ret.append(v)
    for l in ret:
        assert len(l) == 33
    return ret

def euclidian_distance(p1, p2, default=1) -> float:
    if not p1 or not p2:
        return default
    return ((p1[0] - p2[0])**2 + (p1[0] - p2[1])**2)**0.5

def distance_from_camera(poses_x_y: list[list[tuple]]):
    return [1/max([euclidian_distance(joints[torso[i-1]], joints[torso[i]])**2
                   for i in range(len(torso))]) for joints in poses_x_y]

def camera_cut(poses_x_y):
    threshold = 0.6
    ret = [max(euclidian_distance(prev[idx], curent[idx], default=0) for idx in torso + face)
        for prev, curent in zip(poses_x_y, poses_x_y[1:])]
    print(ret)
    ret = [threshold < i for i in ret]
    return ret

def score_handmovement(pose_results, normalize=True):
    hands = left_hand + right_hand
    poses_x_y = x_y_from_poses(pose_results)
    dist_camera = distance_from_camera(poses_x_y)
    threshold = 1.5
    camera_cuts = [threshold > i for i in dist_camera]
    movement = []
    for prev, curent, cut, dist in zip(poses_x_y, poses_x_y[1:], camera_cuts, dist_camera):
        if not cut:
            m = max([euclidian_distance(prev[idx], curent[idx], default=0) for idx in hands])
            if normalize:
                m /= dist
        else:
            m = None
        movement.append(m)
    return movement, dist_camera, camera_cuts

def movement(video_file_path, output_file_path="output.mp4"):
    poses = get_poses(video_file_path, show=False, output_file_path=output_file_path)
    scores, _dist_camera, _cuts = score_handmovement(poses, normalize=False)
    return {
        "movementScores": scores
    }