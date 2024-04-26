import cv2
import numpy as np
import time

def LoadVideo(path):
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        print("Error: Video did not open correctly.")
        return
    
    return vid

def VisualizeFlow(frame, flow, new_frame, threshold):
    #print(frame.shape)
    h, w, c = frame.shape
    step = 10
    y, x = np.mgrid[step//2:h:step, step//2:w:step]
    points = np.column_stack((x.ravel(), y.ravel()))

    color = (0,255,0)

    for i, (x, y) in enumerate(points):
        flow_vectors = flow[y,x]
        #print(flow_vectors)
        dy, dx = flow_vectors

        # Compute the magnitude of the flow vector
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        if magnitude > threshold:
            cv2.arrowedLine(new_frame, (x, y), (int(x+dx), int(y+dy)), color, 1)

    new_frame = cv2.resize(new_frame, (1200,720))

    cv2.imshow('Optical Flow', new_frame)
    cv2.waitKey(1)

def ComputeSpatialCoherence(magnitude, treshhold):
    coherence_map = np.zeros_like(magnitude, dtype=np.uint8)
    
    for x in range(1, magnitude.shape[0] - 1):
        for y in range(1, magnitude.shape[1] - 1):
            neighbor_magnitude = magnitude[x-1:x+2, y-1:y+2]
            coherence = np.mean(neighbor_magnitude) / (magnitude[x, y] + 1e-5)
            coherence_map[x, y] = 255 if coherence > treshhold else 0

    return coherence_map

def ComputeDirectionConsistency(dir_threshold, flow):
    angle = np.arctan2(flow[..., 1], flow[..., 0])

    dir_map = np.zeros_like(angle, dtype=np.uint8)
    for x in range(angle.shape[0]):
        for y in range(angle.shape[1]):
            neighbor_angles = angle[max(0, x-1):min(angle.shape[0], x+2),
                                    max(0, y-1):min(angle.shape[1], y+2)]
            consistency = np.abs(neighbor_angles - angle[x, y]) < dir_threshold
            dir_map[x, y] = 255 if np.all(consistency) else 0

    return dir_map

def ComputeOpticalFlowFromVideo(path):
    vid = LoadVideo(path)

    b, frame = vid.read()
    frame = cv2.resize(frame, (864,486))
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow_accumulator = []

    while True:

        b, new_frame = vid.read()
        new_frame = cv2.resize(new_frame, (1280,720))
        if not b:
            break

        gnew_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Gaussian Blurring
        gframe = cv2.GaussianBlur(gframe, (11,11), 1.0)
        gnew_frame = cv2.GaussianBlur(gnew_frame, (11,11), 1.0)
        
        start_time = time.time()
        flow = cv2.calcOpticalFlowFarneback(gframe, gnew_frame, None, 0.5, 3, 30, 5, 7, 1.5, 0)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution Time:", execution_time, "seconds")

        # Temporal Consistency
        flow_accumulator.append(flow)
        """if len(flow_accumulator) > 1:
            valid_motion_indices = []
            for i in range(len(flow_accumulator)):
                flow_current = flow_accumulator[i]
                magnitude = np.sqrt(flow_current[..., 0]**2 + flow_current[..., 1]**2)
                
                # Thresholding
                #threshold = 3.0
                #motion_indices = np.where(magnitude > threshold)
                
                # Spatial Coherence
                coherence_map = ComputeSpatialCoherence(magnitude, 0.5)
                motion_indices_sc = np.where(coherence_map > 0)

                # Direction Consistency
                dir_map = ComputeDirectionConsistency(np.pi/4, flow_current)
                motion_indices_dc = np.where(dir_map > 0)

                #print(motion_indices_sc)
                #print(motion_indices_dc)

                #combined_motion_indices = np.logical_and(motion_indices_sc, motion_indices_dc)

                #valid_motion_indices.extend(combined_motion_indices)"""

        VisualizeFlow(frame, flow, new_frame, 4.0)
        
        frame = new_frame

    vid.release()


video = 'clip.mp4'
print("Starting Computation")
start_time = time.time()
ComputeOpticalFlowFromVideo(video)
end_time = time.time()
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")
    