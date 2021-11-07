from siamese.prod.predictor import SiamesePredictorOnnx
from siamese.prod.utils import *
import fire

def predict(weight_onnx: str = './weights/siamese-10.onnx', 
            data_dir: str =  '/home/gaungalif/workspace/datasets/omniglot/Alphabet_of_the_Magi', 
            pic_idx: int = 10):

    model = SiamesePredictorOnnx(weight=weight_onnx)
    data_dir = '/home/gaungalif/workspace/datasets/omniglot/Alphabet_of_the_Magi'

    main_imgs, comp_imgs, labels = image_loader(idx=2,data_dir=data_dir)
    model.predict(main_imgs, comp_imgs, labels, pic_idx)

if __name__ == '__main__':
    fire.Fire(predict)