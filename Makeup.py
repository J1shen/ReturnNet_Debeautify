import cv2,imutils,dlib
import numpy as np
#from utils import face_thin_auto, SharpenImage

class Organ():
    def __init__(self, img, img_h,landmark, name, ksize=None):
        '''
        五官部位类
        '''
        self.im_bgr, self.im_hsv, self.landmark, self.name = img, img_h, landmark, name
        self.get_rect()
        self.shape = (int(self.bottom-self.top), int(self.right-self.left))
        self.size = self.shape[0]*self.shape[1]*3
        self.move = int(np.sqrt(self.size/3)/20)
        self.ksize = self.get_ksize()
        self.patch_bgr = self.get_patch(self.im_bgr)
        self.patch_hsv = self.get_patch(self.im_hsv)
        self.patch_mask = self.get_mask_re()
        pass

    def get_ksize(self, rate=15):
        size = max([int(np.sqrt(self.size/3)/rate), 1])
        size = (size if size % 2 == 1 else size+1)
        return (size, size)

    def get_rect(self):
        '''
        获得定位方框
        '''
        ys, xs = self.landmark[:, 1], self.landmark[:, 0]
        self.top, self.bottom, self.left, self.right = np.min(
            ys), np.max(ys), np.min(xs), np.max(xs)

    def get_patch(self, im):
        '''
        截取局部切片
        '''
        shape = im.shape
        x = im[np.max([self.top-self.move, 0]):np.min([self.bottom+self.move, shape[0]]),
               np.max([self.left-self.move, 0]):np.min([self.right+self.move, shape[1]])]
        return x

    def _draw_convex_hull(self, im, points, color):
        '''
        勾画多凸边形
        '''
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def get_mask_re(self, ksize=None):
        '''
        获得局部相对坐标遮罩
        '''
        if ksize == None:
            ksize = self.ksize
        landmark_re = self.landmark.copy()
        landmark_re[:, 1] -= np.max([self.top-self.move, 0])
        landmark_re[:, 0] -= np.max([self.left-self.move, 0])
        mask = np.zeros(self.patch_bgr.shape[:2], dtype=np.float64)

        self._draw_convex_hull(mask,
                               landmark_re,
                               color=1)

        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        return cv2.GaussianBlur(mask, ksize, 0)[:]

    def get_mask_abs(self, ksize=None):
        '''
        获得全局绝对坐标遮罩
        '''
        if ksize == None:
            ksize = self.ksize
        mask = np.zeros(self.im_bgr.shape, dtype=np.float64)
        patch = self.get_patch(mask)
        patch[:] = self.patch_mask[:]
        return mask

    def whitening(self, rate=0.15):
        '''
        提亮美白
        '''
        self.patch_hsv[:, :, -1] = np.minimum(self.patch_hsv[:, :, -1]+self.patch_hsv[:, :, -1]
                                                  * self.patch_mask[:, :, -1]*rate, 255).astype('uint8')
        self.im_bgr[:] = cv2.cvtColor(self.im_hsv, cv2.COLOR_HSV2BGR)[:]


    def brightening(self, rate=0.3):
        '''
        提升鲜艳度
        '''
        patch_mask = self.get_mask_re((1, 1))
        patch_new = self.patch_hsv[:, :, 1]*patch_mask[:, :, 1]*rate
        patch_new = cv2.GaussianBlur(patch_new, (3, 3), 0)
        self.patch_hsv[:, :, 1] = np.minimum(
            self.patch_hsv[:, :, 1]+patch_new, 255).astype('uint8')
        self.im_bgr[:] = cv2.cvtColor(self.im_hsv, cv2.COLOR_HSV2BGR)[:]

    def smooth(self, rate=0.6, ksize=(7, 7)):
        '''
        磨皮
        '''
        if ksize == None:
            ksize = self.get_ksize(80)
        index = self.patch_mask > 0

        patch_new = cv2.GaussianBlur(cv2.bilateralFilter(
            self.patch_bgr, 3, *ksize), ksize, 0)
        self.patch_bgr[index] = np.minimum(
            rate*patch_new[index]+(1-rate)*self.patch_bgr[index], 255).astype('uint8')
        self.im_hsv[:] = cv2.cvtColor(self.im_bgr, cv2.COLOR_BGR2HSV)[:]


    def sharpen(self, rate=0.3):
        '''
        锐化
        '''
        patch_mask = self.get_mask_re((3, 3))
        kernel = np.zeros((9, 9), np.float32)
        kernel[4, 4] = 2.0  # Identity, times two!
        # Create a box filter:
        boxFilter = np.ones((9, 9), np.float32) / 81.0

        # Subtract the two:
        kernel = kernel - boxFilter
        index = patch_mask > 0

        sharp = cv2.filter2D(self.patch_bgr, -1, kernel)
        self.patch_bgr[index] = np.minimum(
            ((1-rate)*self.patch_bgr)[index]+sharp[index]*rate, 255).astype('uint8')
            
class Forehead(Organ):
    def __init__(self, im_bgr, im_hsv,  landmark, mask_organs, name, ksize=None):
        self.mask_organs = mask_organs
        super(Forehead, self).__init__(im_bgr, im_hsv,landmark, name, ksize)

    def get_mask_re(self, ksize=None):
        '''
        获得局部相对坐标遮罩
        '''
        if ksize == None:
            ksize = self.ksize
        landmark_re = self.landmark.copy()
        landmark_re[:, 1] -= np.max([self.top-self.move, 0])
        landmark_re[:, 0] -= np.max([self.left-self.move, 0])
        mask = np.zeros(self.patch_bgr.shape[:2], dtype=np.float64)

        self._draw_convex_hull(mask,
                               landmark_re,
                               color=1)

        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))

        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        patch_organs = self.get_patch(self.mask_organs)
        mask = cv2.GaussianBlur(mask, ksize, 0)[:]
        mask[patch_organs > 0] = (1-patch_organs[patch_organs > 0])
        return mask

class Face(Organ):
    '''
    脸类
    '''
    def __init__(self, im_bgr, img_hsv, landmarks, index):
        self.index = index
        # 五官名称
        self.organs_name = ['jaw', 'mouth', 'nose',
                            'left eye', 'right eye', 'left brow', 'right brow']

        # 五官等标记点
        self.organs_points = [list(range(0, 17)), list(range(48, 61)), list(range(27, 35)), list(
            range(42, 48)), list(range(36, 42)), list(range(22, 27)), list(range(17, 22))]

        # 实例化脸对象和五官对象
        self.organs = {name: Organ(im_bgr, img_hsv, landmarks[points], name) for name, points in zip(
            self.organs_name, self.organs_points)}

        # 获得额头坐标，实例化额头
        mask_nose = self.organs['nose'].get_mask_abs()
        mask_organs = (self.organs['mouth'].get_mask_abs()+mask_nose+self.organs['left eye'].get_mask_abs(
        )+self.organs['right eye'].get_mask_abs()+self.organs['left brow'].get_mask_abs()+self.organs['right brow'].get_mask_abs())
        forehead_landmark = self.get_forehead_landmark(
            im_bgr, landmarks, mask_organs, mask_nose)
        self.organs['forehead'] = Forehead(
            im_bgr, img_hsv, forehead_landmark, mask_organs, 'forehead')
        mask_organs += self.organs['forehead'].get_mask_abs()

        # 人脸的完整标记点
        self.FACE_POINTS = np.concatenate([landmarks, forehead_landmark])
        super(Face, self).__init__(im_bgr, img_hsv, self.FACE_POINTS, 'face')

        mask_face = self.get_mask_abs()-mask_organs
        self.patch_mask = self.get_patch(mask_face)
        pass

    def get_forehead_landmark(self, im_bgr, face_landmark, mask_organs, mask_nose):
        '''
        计算额头坐标
        '''
        # 画椭圆
        radius = (np.linalg.norm(
            face_landmark[0]-face_landmark[16])/2).astype('int32')
        center_abs = tuple(
            ((face_landmark[0]+face_landmark[16])/2).astype('int32'))

        angle = np.degrees(np.arctan(
            (lambda l: l[1]/l[0])(face_landmark[16]-face_landmark[0]))).astype('int32')
        mask = np.zeros(mask_organs.shape[:2], dtype=np.float64)
        cv2.ellipse(mask, center_abs, (radius, radius), angle, 180, 360, 1, -1)
        # 剔除与五官重合部分
        mask[mask_organs[:, :, 0] > 0] = 0
        # 根据鼻子的肤色判断真正的额头面积
        index_bool = []
        for ch in range(3):
            mean, std = np.mean(im_bgr[:, :, ch][mask_nose[:, :, ch] > 0]), np.std(
                im_bgr[:, :, ch][mask_nose[:, :, ch] > 0])
            up, down = mean+0.5*std, mean-0.5*std
            index_bool.append((im_bgr[:, :, ch] < down)
                              | (im_bgr[:, :, ch] > up))
        index_zero = (
            (mask > 0) & index_bool[0] & index_bool[1] & index_bool[2])
        mask[index_zero] = 0
        index_abs = np.array(np.where(mask > 0)[::-1]).transpose()
        landmark = cv2.convexHull(index_abs).squeeze()
        return landmark

class Making_up():
    def __init__(self, predictor_path="/content/opencv-pyqt-makeup-software/data/shape_predictor_68_face_landmarks.dat"):
        self.photo_path = []
        self.PREDICTOR_PATH = predictor_path
        self.faces = []
        # 人脸定位、特征提取器，来自dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.im_bgr = []
        self.im_hsv = []
        self.rect = []


    def read_img(self, fname, scale=1):
        '''
        读取图片
        '''
        img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
        self.im_bgr = imutils.resize(img, width=600)
        if type(self.im_bgr ) == type(None):
            raise ValueError(
                'Opencv error reading image "{}" , got None'.format(fname))
        self.im_hsv = cv2.cvtColor(self.im_bgr , cv2.COLOR_BGR2HSV)
        rects = self.detector(self.im_bgr , 1)
        if len(rects) < 1:
            print('no face detected')
            return
        
        self.faces = [Face(self.im_bgr ,self.im_hsv ,np.array([[p.x, p.y] for p in self.predictor(self.im_bgr , rect).parts()]), i) for i, rect in enumerate(rects)]

        return self.im_bgr
    
    def _mapfaces(self, fun, value):
        '''
        对每张脸进行迭代操作
        '''
        for face in self.faces:
            fun(face, value)

    def _Laplace(self,val):
        value = min(1, max(val, 0))
        kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
        print('-[INFO] laplace:', value)
        self.img_bgr = SharpenImage(self.img_bgr)
        # self.temp_bgr = cv2.filter2D(self.temp_bgr, -1, kernel)
        self.img_bgr = np.minimum(self.img_bgr, 255).astype('uint8')

    def _Thin(self, val):
        value = min(1, max(val, 0))
        print('-[INFO] thin:', value)
        self.img_bgr = face_thin_auto(self.img_bgr, self.detector, self.predictor)

    def _sharpen(self,val):
        value = min(1, max(val, 0))
        print('-[INFO] sharpen:', value)

        def fun(face, value):
            face.organs['left eye'].sharpen(value, confirm=False)
            face.organs['right eye'].sharpen(value, confirm=False)
        self._mapfaces(fun, value)

    def _whitening(self, val):
        value = min(1, max(val, 0))
        print('-[INFO] whitening:', value)

        def fun(face,value):
            face.organs['left eye'].whitening(value)
            face.organs['right eye'].whitening(value)
            face.organs['left brow'].whitening(value)
            face.organs['right brow'].whitening(value)
            face.organs['nose'].whitening(value)
            face.organs['forehead'].whitening(value)
            face.organs['mouth'].whitening(value)
            face.whitening(value)
        self._mapfaces(fun, value)

    def _brightening(self, val):
        value = min(1, max(val, 0))
        print('-[INFO] brightening:', value)

        def fun(face, value):
            face.organs['mouth'].brightening(value)
        self._mapfaces(fun, value)

    def _smooth(self,val):
        value = min(1, max(val, 0))
        print('-[INFO] smooth:', value)

        def fun(face, value):
            face.smooth(value)
            face.organs['nose'].smooth(value*2/3)
            face.organs['forehead'].smooth(value*3/2)
            face.organs['mouth'].smooth(value)
        self._mapfaces(fun, value)
