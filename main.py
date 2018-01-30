from data.DataSymbol_Iso import usefullTools
from python.recognizer import segmenter
from python.recognizer import TrainMLP

if __name__ == '__main__':
    # segmenter.main(["list_files_TestEM2014.txt"])  # To do once to generate .LG files.
    # usefullTools.main()  # To do to generate data .npz, supposing you have generated images for each dataset.
    TrainMLP.main()

