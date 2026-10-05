import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('F:/Desktop/Benchmark/verify1008/RTDETR-main/runs/train/20260128/FCA-IRSTD1K/weights/best.pt') # select your model.pt path
    model.predict(source='D:/Datasets/IRSTD-1k-YOLO/train/images',
                  conf=0.25,
                  project='H:/visual',
                  name='HWRTDETR-IRSTD-1k',
                  save=True,
                  # visualize=True # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )