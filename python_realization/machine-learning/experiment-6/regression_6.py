import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy import conj, real


class HOG():
    def __init__(self, winSize):
        self.winSize = winSize  # 窗⼝⼤⼩
        self.blockSize = (8, 8)  # 块⼤⼩
        self.blockStride = (4, 4)   # 块在窗⼝中滑动的步⻓
        self.cellSize = (4, 4)  # 块中的单元（cell）⼤⼩
        self.nbins = 9  # ⽅向梯度直⽅图的柱数
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nbins)

    def get_feature(self, image):
        winStride = self.winSize
        hist = self.hog.compute(image, winStride, padding=(0, 0))
        w, h = self.winSize
        sw, sh = self.blockStride
        w = w // sw - 1
        h = h // sh - 1
        return hist.reshape(w, h, 36).transpose(2, 1, 0)


class Tracker():
    def __init__(self):
        self.max_patch_size = 256
        self.padding = 2.5
        self.lambdar = 0.0001  # 岭回归的正则化参数
        self.update_rate = 0.012  # 学习率

    def get_feature(self, image, roi):
        """
        处理图像并获取特征
        """
        cx, cy, w, h = roi
        # padding让样本中含有需要学习的背景信息，同时保证样本中⽬标的完整性
        w = int(w * self.padding) // 2 * 2
        h = int(h * self.padding) // 2 * 2
        x = int(cx - w // 2)
        y = int(cy - h // 2)
        sub_image = image[y:y+h, x:x+w, :]
        resized_image = cv2.resize(sub_image, (self.pw, self.ph))
        # 提取HOG特征
        feature = self.hog.get_feature(resized_image)
        fc, fh, fw = feature.shape
        self.scale_h = float(fh) / h
        self.scale_w = float(fw) / w
        # 针对提取得到的特征，采⽤余弦窗进⾏相乘平滑计算
        # 因为移动样本的边缘⽐较突⺎，会⼲扰训练的结果
        # 如果加了余弦窗，图像边缘像素值就都接近0了，循环移位过程中只要⽬标保持完整那这个样本就是合理的
        hann2t, hann1t = np.ogrid[0:fh, 0:fw]
        hann1t = 0.5 * (1 - np.cos(2*np.pi*hann1t / (fw-1)))
        hann2t = 0.5 * (1 - np.cos(2*np.pi*hann2t / (fh-1)))
        hann2d = hann2t * hann1t
        feature = feature * hann2d
        feature = np.sum(feature, axis=0)
        return feature

    def gaussian_peak(self, w, h):
        """
        使⽤⾼斯函数制作样本标签y
        """
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h // 2, w // 2  # ⽬标框中⼼点
        y, x = np.mgrid[-syh:-syh+h, -sxh:-sxh+w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        # ⽣成标签(h,w)，越靠近中⼼点值越⼤
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x**2 + y**2)/(2. * sigma**2)))
        return g

    def train(self, x, y, lambdar):
        # TODO: 实现回归求解的代码，返回 W_hat
        x_hat = fft2(x)
        y_hat = fft2(y)
        x_c = conj(x_hat)
        W_hat = (x_c*y_hat)/((x_c*x_hat)+lambdar)
        return W_hat

    def detect(self, W_hat, z):
        # TODO: 实现⽬标在新帧中检测的代码，返回 f(z)
        z_hat = fft2(z)
        W = real(ifft2(W_hat*z_hat))
        return W

    def init(self, image, roi):
        x1, y1, w, h = roi
        # ⽬标区域的中⼼坐标
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)
        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))
        x = self.get_feature(image, roi)
        y = self.gaussian_peak(x.shape[1], x.shape[0])
        self.wf = self.train(x, y, self.lambdar)  # 求解权重参数w
        self.x = x
        self.roi = roi

    def update(self, image):
        """
        对给定的图像，重新计算其⽬标的位置
        """
        cx, cy, w, h = self.roi
        max_response = -1
        # 尝试多个尺度，也就是在⽬标跟踪的过程中，适应⽬标⼤⼩的变化
        for scale in [0.85, 1.0, 1.02]:
            roi = map(int, (cx, cy, w * scale, h * scale))
            z = self.get_feature(image, roi)
            responses = self.detect(self.wf, z)  # 检测⽬标
            height, width = responses.shape
            idx = np.argmax(responses)
            res = np.max(responses)
            # 选取检测预测值最⼤的位置作为⽬标的新位置
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z
        # 更新,每次训练得到的参数受以往训练得到的参数的影响，有⼀个加权的过程
        self.roi = (cx + dx, cy + dy, best_w, best_h)
        # self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_peak(best_z.shape[1], best_z.shape[0])
        new_w = self.train(best_z, y, self.lambdar)
        self.wf = self.wf * (1 - self.update_rate) + new_w * self.update_rate
        cx, cy, w, h = self.roi
        # 返回⽬标区域的中⼼坐标和⼤⼩
        return (cx - w // 2, cy - h // 2, w, h)


if __name__ == '__main__':
    # 打开视频⽂件
    cap = cv2.VideoCapture('car.avi')
    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 设置播放速度
    new_fps = 0.5*fps
    cap.set(cv2.CAP_PROP_FPS, new_fps)
    # 读取第⼀帧
    ok, frame = cap.read()
    if not ok:
        print("error reading video")
        exit(-1)
    roi = cv2.selectROI("tracking", frame, False, False)
    # 创建跟踪器对象
    tracker = Tracker()
    # 初始化跟踪器
    tracker.init(frame, roi)
    # 开始跟踪
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        # 更新跟踪器
        x, y, w, h = tracker.update(frame)
        # 绘制跟踪结果框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
