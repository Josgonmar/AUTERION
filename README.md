# AUTERION: (AU)tomatic (TE)xt (R)ecognition and translat(ION)
Recognize and translate text from any image. You can both specify the original language of the input text or let the API detect it. Of course, setting the input language usually throws better results.

## Dependencies:
* [Python](https://www.python.org/doc/) - 3.10.5
* [OpenCV](https://docs.opencv.org/4.6.0/) - 4.6.0
* [Numpy](https://numpy.org/doc/stable/) - 1.22.4
* [Googletrans](https://py-googletrans.readthedocs.io/en/latest/) - 3.0.0
* [Streamlit](https://docs.streamlit.io/library/get-started) - 1.10.0 (Only required to run the streamlit version)

## How to use:
1. Copy all the images inside the */visuals* folder. They will be translated one by one automatically as a console message.

2. Go to the *src* folder and execute `AUTERION.py`
```console
    $ python AUTERION.py
```
3. You will be asked to enter the input language (if known), and the language to translate into.

*Note: You have to write the language as it's written in the Google API, otherwise it won't be recognized. Check the supported languages [here.](https://py-googletrans.readthedocs.io/en/latest/)*

## License:
Feel free to use this programa whatever you like!
