from utils import readVideo,videoWriter
from yolov3 import CCPD_YOLO_Detector,plot_one_box
import torch
from facenet_pytorch import MTCNN

if __name__ == '__main__':
    device = torch.device('cpu')
    # detector = CCPD_YOLO_Detector(device=device)

    mtcnn=MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        prewhiten=True,
        select_largest=True,#True,boxes按照面积的大小降序排列
        keep_all=True,
        device=device
    )

    cap,videoinfo= readVideo('data/laofoye_high.avi')
    print(videoinfo)
    cap_writer = videoWriter('hello.avi',scale=(videoinfo['width'],videoinfo['height']),fps=3)

    cnt = 0
    rate = 10
    buf = []
    pos = 0
    while True:
        retval, frame = cap.read()
        if not retval: break
        pos += 1

        if pos % rate == 0:
            # buf = np.stack(buf, axis=0)
            #         newframe=np.mean(buf,axis=0).astype(np.uint8)
            newframe = buf[-1]

            ###############目标检索########################
            # result = detector.predict(newframe)
            # for x1, y1, x2, y2, conf, cls in result:
            #     plot_one_box((x1, y1, x2, y2), newframe, label='%.2f' % conf, color=(0, 255, 0))

            ###############人脸检索#####################
            newframe = newframe[:, :, ::-1]
            faces,boxes = mtcnn(newframe)
            if faces is not None:
                print(boxes)
                newframe=newframe[:, :, ::-1]
                for x1, y1, x2, y2 in boxes:
                    plot_one_box((int(x1), int(y1), int(x2), int(y2)), newframe, label='P', color=(0, 255, 0))
                newframe=newframe[:, :, ::-1]
            ###########################################
                newframe=newframe[:, :, ::-1]
                cap_writer.write(newframe)
            buf = []
        else:
            buf.append(frame)
    cap.release()
    cap_writer.release()
    