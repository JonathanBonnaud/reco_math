import os

from python.code import convertInkmlToImg
from python.code import usefulTools
from python.recognizer import segmenter
from python.recognizer import TrainMLP
from python.recognizer import write_pred_to_lg

if __name__ == '__main__':
    # segmenter.main(["list_files_TestEM2014.txt"])  # To do once to generate first .LG files + .inkml of hypothesis.
    # os.system('python3 python/code/convertInkmlToImg.py LGs/segments_hypo 28 2 LGs/img_segments_hypo/')
    # usefulTools.main()  # To do to generate data .npz, supposing you have generated images for each dataset.
    TrainMLP.main()
    write_pred_to_lg.write("PRED.txt")
