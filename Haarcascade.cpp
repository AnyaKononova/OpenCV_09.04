#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    CascadeClassifier faceCascade, eyesCascade, smileCascade;
    if (!faceCascade.load("haarcascade_frontalface_alt.xml") ||
        !eyesCascade.load("haarcascade_eye_tree_eyeglasses.xml") ||
        !smileCascade.load("haarcascade_smile.xml")) {
        cout << "Error" << endl;
        return -1;
    }

    string videoPath = "C:/Users/Nuta/Documents/Open CV/09.04/task1/task1/video.mp4";

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cout << "Error!" << endl;
        return -1;
    }

    string outputVideoPath = "C:/Users/Nuta/Documents/Open CV/09.04/task1/task1/output_video.avi";
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter videoWriter(outputVideoPath, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height));

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cout << "End" << endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(frame, faces[i], Scalar(255, 0, 0), 2);
            Mat faceROI = gray(faces[i]);
            vector<Rect> eyes;
            eyesCascade.detectMultiScale(faceROI, eyes);

            for (size_t j = 0; j < eyes.size(); j++) {
                Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
                circle(frame, eye_center, radius, Scalar(0, 255, 0), 2);
            }

            vector<Rect> smiles;
            smileCascade.detectMultiScale(faceROI, smiles, 1.165, 35, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

            for (size_t k = 0; k < smiles.size(); k++) {
                rectangle(frame, Point(faces[i].x + smiles[k].x, faces[i].y + smiles[k].y),
                    Point(faces[i].x + smiles[k].x + smiles[k].width, faces[i].y + smiles[k].y + smiles[k].height),
                    Scalar(0, 0, 255), 2);
            }
        }
        videoWriter.write(frame);

        imshow("Face Detection", frame);

        if (waitKey(25) == 'q') {
            break;
        }
    }

    return 0;
}
