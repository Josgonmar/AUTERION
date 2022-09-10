from cgitb import text
import cv2
import numpy as np
import googletrans
import streamlit as st

class Params:
    class TextDetector:
        path = "../../resources/DB_TD500_resnet50.onnx"
        input_size = (640, 640)
        bin_thresh = 0.3
        poly_thresh = 0.5
        mean = (122.67891434, 116.66876762, 104.00698793)
        scale = 1.0/255
    
    class TextRecognizer:
        path = "../../resources/crnn_cs.onnx"
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
        self.__loadVocabulary()
        self.__loadModels()

        st.title('(AU)tomatic (TE)xt (R)ecognition and translat(ION)')
    
    def run(self):

        uploaded_image_file = st.file_uploader("Upload an image:", type=['png', 'jpg'])

        if uploaded_image_file is not None:
            image = self.__toOpenCV(uploaded_image_file)
            input_col, output_col = st.columns(2)

            autodetect = st.checkbox('Autodetect the source language', value=False)

            src_lang = st.selectbox('Select source language:', googletrans.LANGUAGES.items(), disabled=autodetect)
            dst_lang = st.selectbox('Select language to translate into:', googletrans.LANGUAGES.items())

            with input_col:
                st.header('Original input')
                st.image(image, channels='BGR', use_column_width=True)

            boxes, confs = self.__text_detector.detect(image)
            
            words = []
            for box in boxes[::-1]:
                txt_roi = self.__rotateText(box, image)
                recognized_text = self.__text_recognizer.recognize(txt_roi)

                if autodetect:
                    translation = self.__translator.translate(recognized_text, dst_lang[0])
                else:
                    translation = self.__translator.translate(recognized_text, dst_lang[0], src_lang[0])

                words.append(translation.text)
            
            text = words[0]
            for word in words[1:]:
                text += ' ' + word

            with output_col:
                st.header('Translation')
                st.text(text)

    def __rotateText(self, box, image):
        input_box = np.asarray(box).astype(np.float32)
        req_size = self.__params.TextDetector().input_size
        target_box = np.array([[0,req_size[1]-1], [0,0], [req_size[0]-1,0], [req_size[0]-1,req_size[1]-1]], dtype='float32')
        rot_matrix = cv2.getPerspectiveTransform(input_box, target_box)

        return cv2.warpPerspective(image, rot_matrix, req_size)

    def __loadVocabulary(self):
        f = open('../../resources/alphabet_94.txt')
        for letter in f:
            self.__vocabulary.append(letter.strip())

        f.close()

    def __toOpenCV(self, src):
        raw_bytes = np.asarray(bytearray(src.read()), dtype=np.uint8)
        dst = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

        return dst
    
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
    AUTERION_obj.run()