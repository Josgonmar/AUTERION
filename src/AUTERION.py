import cv2
import numpy as np
import os
import googletrans

class Params:
    class TextDetector:
        path = "../resources/DB_TD500_resnet50.onnx"
        input_size = (640, 640)
        bin_thresh = 0.3
        poly_thresh = 0.5
        mean = (122.67891434, 116.66876762, 104.00698793)
        scale = 1.0/255
    
    class TextRecognizer:
        path = "../resources/crnn_cs.onnx"
        decode_type = "CTC-greedy"
        input_size = (100,32)
        mean = (127.5, 127.5, 127.5)
        scale = 1/127.5

class AUTERION:
    __images = []
    __vocabulary = []
    __params = Params()
    __translator = None
    __text_detector = None
    __text_recognizer = None

    def __init__(self):
        self.__loadImages()
        self.__loadVocabulary()
        self.__loadModels()
    
    def run(self, dst_lang='en', src_lang=''):

        if len(self.__images) != 0:

            for img in self.__images:

                print('[INFO] Translating image:')

                boxes, confs = self.__text_detector.detect(img)

                try:
                    for box in boxes[::-1]:
                        txt_roi = self.__rotateText(box, img)
                        recognized_text = self.__text_recognizer.recognize(txt_roi)

                        if src_lang:
                            translation = self.__translator.translate(recognized_text, dst_lang, src_lang)
                        else:
                            translation = self.__translator.translate(recognized_text, dst_lang)

                        print('[INFO] Recognized Text[{}]: {} -> Translated Text[{}]: {}'.format(
                            googletrans.LANGUAGES[translation.src],
                            recognized_text,
                            googletrans.LANGUAGES[dst_lang],
                            translation.text))
                except:
                    print('[ERROR] Source language does not match')
        else:
            print('[INFO] No images loaded. Finishing the process...')

    def __rotateText(self, box, image):
        input_box = np.asarray(box).astype(np.float32)
        req_size = self.__params.TextDetector().input_size
        target_box = np.array([[0,req_size[1]-1], [0,0], [req_size[0]-1,0], [req_size[0]-1,req_size[1]-1]], dtype='float32')
        rot_matrix = cv2.getPerspectiveTransform(input_box, target_box)

        return cv2.warpPerspective(image, rot_matrix, req_size)

    def __loadImages(self):
        try:
            for file in os.listdir('../visuals'):
                self.__images.append(cv2.imread('../visuals/' + file, cv2.IMREAD_COLOR))
            
            if len(self.__images) == 0:
                print('[INFO] No image(s) in folder')
            else:
                print('[INFO] Loaded image(s)')
        except:
            print('[ERROR] An error occured while reading the image(s)')

    def __loadVocabulary(self):
        f = open('../resources/alphabet_94.txt')
        for letter in f:
            self.__vocabulary.append(letter.strip())

        f.close()
    
    def __loadModels(self):
        self.__translator = googletrans.Translator()

        self.__text_detector = cv2.dnn_TextDetectionModel_DB(self.__params.TextDetector().path)
        self.__text_detector.setBinaryThreshold(self.__params.TextDetector().bin_thresh).setPolygonThreshold(self.__params.TextDetector().poly_thresh)
        self.__text_detector.setInputParams(self.__params.TextDetector().scale, self.__params.TextDetector().input_size, self.__params.TextDetector().mean, True)

        self.__text_recognizer = cv2.dnn_TextRecognitionModel(self.__params.TextRecognizer().path)
        self.__text_recognizer.setDecodeType(self.__params.TextRecognizer().decode_type)
        self.__text_recognizer.setVocabulary(self.__vocabulary)
        self.__text_recognizer.setInputParams(self.__params.TextRecognizer().scale, self.__params.TextRecognizer().input_size, self.__params.TextRecognizer().mean, True)

if __name__ == '__main__':
    AUTERION_obj = AUTERION()
    src_lang = input('Enter the original text language if known. Leave empty if it is unknown: ')
    dst_lang = input('Enter the language to translate into: ')
    AUTERION_obj.run(dst_lang, src_lang)