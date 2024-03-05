class Image():
    def __init__(self, path):

        self._original_image = cv2.resize(cv2.imread(f'{path}'), (320, 170), cv2.INTER_LINEAR)
        self._gray_scale_image = cv2.cvtColor(self._original_image, cv2.COLOR_BGR2GRAY)
        self._image_size = None
        self._image_ifft = None
    

    @property
    def original_image(self):
        return self._original_image
    
    @property
    def gray_scale_image(self):
        return self._gray_scale_image
    
    @property
    def image_size(self):
        return self._image_size
    

    @image_size.setter
    def image_size(self, value):
        self._original_image = cv2.resize(self._original_image, value)
        self._image_size = value