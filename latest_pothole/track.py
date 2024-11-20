import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import os

def run_tracking(start_x, start_y, end_x, end_y, video_path, count=0, offset=7):
    model = YOLO(r"C:\Users\lokes\Downloads\P_pothole\latest_pothole\latestptt.pt")
    class_list = ['potholes']
    down = {}
    counter_down = set()

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create the "results" folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join("results", f"processed_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Create a window to display the processed frames
    cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        results = model.predict(frame)
        detections = results[0].boxes.data.detach().cpu().numpy()
        px = pd.DataFrame(detections).astype("float")
        detected_objects = []
        for _, row in px.iterrows():
            x1, y1, x2, y2, _, class_id = map(int, row)
            class_name = class_list[class_id]
            if class_name == 'potholes':
                detected_objects.append([x1, y1, x2, y2])

        bbox_id = tracker.update(detected_objects)

        for bbox in bbox_id:
            x1, y1, x2, y2, obj_id = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if start_y - offset < cy < start_y + offset:
                down[obj_id] = cy
                if obj_id in down:
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(obj_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    counter_down.add(obj_id)

        text_color = (255, 255, 255)
        red_color = (0, 0, 255)

        # Ensure coordinates are integers
        start_y = int(start_y)
        start_x = int(start_x)
        end_x = int(end_x)

        cv2.line(frame, (start_x, start_y), (end_x, start_y), red_color, 3)
        cv2.putText(frame, 'red line', (start_x, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        downwards = len(counter_down)
        cv2.putText(frame, 'going down - ' + str(downwards), (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1, cv2.LINE_AA)

        # Display the processed frame in its actual size
        cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Video", width, height)
        cv2.imshow("Processed Video", frame)

        # Write the processed frame to the output video
        out.write(frame)

        # Check if the user pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path

#############################################################################################



# import cv2
# import pandas as pd
# from ultralytics import YOLO
# from tracker import Tracker
# import os

# def run_tracking(start_x, start_y, end_x, end_y, video_path, count=0, offset=7):
#     model = YOLO(r"latestptt.pt")
#     class_list = ['potholes']
#     tracker = Tracker()
#     down = {}
#     counter_down = set()

#     cap = cv2.VideoCapture(video_path)

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Create the "results" folder if it doesn't exist
#     os.makedirs("results", exist_ok=True)

#     # Create a VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video_path = os.path.join("results", f"processed_video.mp4")
#     # output_video_path = os.path.join("results", f"processed_{os.path.basename(video_path)}")
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     # Create a window to display the processed frames
#     cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         count += 1

#         results = model.predict(frame)
#         detections = results[0].boxes.data.detach().cpu().numpy()
#         px = pd.DataFrame(detections).astype("float")
#         detected_objects = []
#         for _, row in px.iterrows():
#             x1, y1, x2, y2, _, class_id = map(int, row)
#             class_name = class_list[0]
#             if class_name == 'potholes':
#                 detected_objects.append([x1, y1, x2, y2])

#         bbox_id = tracker.update(detected_objects)

#         for bbox in bbox_id:
#             x1, y1, x2, y2, obj_id = bbox
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

#             cv2.circle(frame,(cx,cy),4,(0,0,255),-1) #draw ceter points of bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
#             cv2.putText(frame,str(obj_id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
#             y = 308
#             offset = 7
#             if y < (cy + offset) and y > (cy - offset):
#             # if start_y - offset < cy < start_y + offset:
#                 down[obj_id] = cy
#                 if obj_id in down:
#                     # cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
#                     # cv2.putText(frame, str(obj_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
#                     counter_down.add(obj_id)
#             # if start_y - offset < cy < start_y + offset:
#             #     down[obj_id] = cy
#             #     if obj_id in down:
#             #         cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
#             #         cv2.putText(frame, str(obj_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
#             #         counter_down.add(obj_id)

#         text_color = (255, 255, 255)
#         red_color = (0, 0, 255)

#         # Ensure coordinates are integers
#         # start_y = int(start_y)
#         # start_x = int(start_x)
#         # end_x = int(end_x)

#         # cv2.line(frame, (start_x, start_y), (end_x, start_y), red_color, 3)
#         # cv2.putText(frame, 'red line', (start_x, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

#         downwards = len(counter_down)
#         cv2.putText(frame, 'Potholes counts - ' + str(downwards), (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1, cv2.LINE_AA)

#         # Display the processed frame in its actual size
#         cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
#         cv2.resizeWindow("Processed Video", width, height)
#         cv2.imshow("Processed Video", frame)

#         # Write the processed frame to the output video
#         out.write(frame)

#         # Check if the user pressed the 'q' key to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     return output_video_path