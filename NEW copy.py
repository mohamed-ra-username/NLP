from time import sleep
from cv2.typing import MatLike
import wx
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Corrected Gaussian function
# def Gaussian(img, ker_size, scaling_factor):
#     print("Gaussian")
#     var = ker_size / 6
#     kernel = np.zeros((ker_size, ker_size), dtype=np.float32)
#     center = ker_size // 2
#     for s in range(ker_size):
#         for t in range(ker_size):
#             r2 = (s - center) ** 2 + (t - center) ** 2
#             kernel[s, t] = scaling_factor * np.exp(-r2 / (2 * var ** 2))
#     print("kernel")
#     kernel /= np.sum(kernel)
#     pad = ker_size // 2
#     padded = np.pad(img, pad, 'reflect')
#     out = np.zeros_like(img, dtype=np.float32)
#     cv2.GaussianBlur(img, (ker_size, ker_size), 0)
#     print("out")
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             out[i, j] = np.sum(padded[i:i+ker_size, j:j+ker_size] * kernel)
#     return out
def Gaussian(img, ker_size, scaling_factor=1):
    if ker_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    
    # Generate Gaussian kernel using OpenCV
    kernel_1d = cv2.getGaussianKernel(ker_size, ker_size / 6)
    kernel = np.outer(kernel_1d, kernel_1d) * scaling_factor  # Create 2D kernel
    
    # Normalize the kernel
    kernel /= np.sum(kernel)
    
    # Apply convolution using OpenCV's filter2D for better performance
    out = cv2.filter2D(img, -1, kernel)
    
    return out

def local_his_eq(img: MatLike, ker_size: int):
    """
    Perform local histogram equalization on patches of an image using CLAHE.

    Args:
        img (MatLike): Input grayscale image.
        ker_size (int): Kernel size for local histogram equalization (tile grid size).

    Returns:
        np.ndarray: Image after local histogram equalization.
    """
    # Ensure the input image is grayscale
    if len(img.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")

    # Create a CLAHE object with the specified kernel size
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ker_size, ker_size))
    
    # Apply CLAHE to the input image
    result = clahe.apply(img)
    
    return result
# Local histogram equalization on patches
# def local_his_eq(img: MatLike, ker_size: int) -> np.ndarray:
#     # Use OpenCV's CLAHE for better performance and simplicity
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ker_size, ker_size))
#     result = clahe.apply(img)
#     return result
# def local_his_eq(img, ker_size):
#     # Create a CLAHE object with the specified kernel size
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ker_size, ker_size))
    
#     # Apply CLAHE to the input image
#     result = clahe.apply(img)
#     result = cv2.equalizeHist(result)
#     # Convert the result back to the original image type
#     return result
# def plot_img_and_hist(img, axes, bins=256):
#     """Plot an image along with its histogram and cumulative histogram."""
#     hist,bins = np.histogram(img.flatten(),256,[0,256])
 
#     cdf = hist.cumsum()
#     cdf_normalized = cdf * float(hist.max()) / cdf.max()
    
#     plt.plot(cdf_normalized, color = 'b')
#     plt.hist(img.flatten(),256,[0,256], color = 'r')
#     plt.xlim([0,256])
#     plt.legend(('cdf','histogram'), loc = 'upper left')
#     plt.show()

class ImageProcessingGUI(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(1000, 800),
                         style=wx.DEFAULT_FRAME_STYLE | wx.RESIZE_BORDER)
        self.original_img = None
        self.proc_img = None
        self.setup_ui()
        self.Show()

    def setup_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Load Image Button placed above the images
        load_btn = wx.Button(self, label="Load Image")
        load_btn.Bind(wx.EVT_BUTTON, self.ask_load_image)
        sizer.Add(load_btn, 0, wx.ALL | wx.CENTER, 10)

        # Image display panels
        disp_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.before_bmp = wx.StaticBitmap(self, size=(340, 340))
        self.after_bmp = wx.StaticBitmap(self, size=(340, 340))

        for lbl, bmp in [("Before", self.before_bmp), ("After", self.after_bmp)]:
            box = wx.StaticBox(self, label=lbl)
            box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
            box_sizer.Add(bmp, 1, wx.EXPAND | wx.ALL, 5)
            disp_sizer.Add(box_sizer, 0, wx.ALL, 5)
        sizer.Add(disp_sizer, 0, wx.CENTER)

        # Buttons grid
        btns = [
            ("Edge Detect", self.edge_detect),
            ("Threshold", self.threshold),
            ("Custom Gaussian Blur", self.custom_gaussian),
            ("Local Histogram Equalization", self.local_hist_eq),
            ("Enhance", self.enhance),
            ("Gray & Enhance", self.gray_enhance),
        ]

        num_buttons = len(btns)
        cols = 2
        rows = (num_buttons + cols - 1) // cols

        btn_sizer = wx.GridSizer(rows, cols, 10, 10)
        for label, handler in btns:
            btn = wx.Button(self, label=label)
            btn.Bind(wx.EVT_BUTTON, handler)
            btn_sizer.Add(btn, 0, wx.EXPAND)
        sizer.Add(btn_sizer, 0, wx.ALL | wx.CENTER, 10)

        # "Back to Original" Button
        self.back_btn = wx.Button(self, label="Back to Original")
        self.back_btn.Bind(wx.EVT_BUTTON, self.back_to_original)
        sizer.Add(self.back_btn, 0, wx.ALL | wx.CENTER, 10)

        self.SetSizer(sizer)

    def display_image(self, img, bmp_widget):
        if img is None:
            return
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img.copy()

        # Resize to fit display
        h, w = img_bgr.shape[:2]
        max_dim = 340
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = max_dim
            new_h = int(max_dim / aspect_ratio)
        else:
            new_h = max_dim
            new_w = int(max_dim * aspect_ratio)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Centered in 340x340 canvas
        canvas = np.zeros((340, 340, 3), dtype=np.uint8)
        y_off = (340 - new_h) // 2
        x_off = (340 - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        bmp = wx.Bitmap.FromBufferRGBA(width, height, cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA).tobytes())
        bmp_widget.SetBitmap(bmp)
        self.Layout()

    def ask_load_image(self, event=None):
        with wx.FileDialog(self, "Open Image", "", "",
                           "Image Files (*.png;*.jpg;*.jpeg)|*.png;*.jpg;*.jpeg",
                           wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                self.load_image(path)
        # Disable back button until an image is loaded
        self.back_btn.Enable(self.original_img is not None)
    def load_image(self, path):
        img = cv2.imread(path)
        if img is not None:
            self.original_img = img
            self.proc_img = None
            self.display_image(self.original_img, self.before_bmp)
            self.display_image(self.original_img, self.after_bmp)
            self.back_btn.Enable(True)
        else:
            wx.MessageBox("Failed to load image.", "Error")
    def get_current_img(self):
        return self.proc_img if self.proc_img is not None else self.original_img

    def back_to_original(self, event):
        if self.original_img is not None:
            self.proc_img = None
            self.display_image(self.original_img, self.after_bmp)

    def edge_detect(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        edges = cv2.Canny(gray, 100, 200)
        self.proc_img = edges
        self.display_image(self.proc_img, self.after_bmp)

    def threshold(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.proc_img = thresh
        self.display_image(self.proc_img, self.after_bmp)

    def custom_gaussian(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        blurred = Gaussian(gray, 43, 1)
        self.proc_img = cv2.cvtColor(blurred.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display_image(self.proc_img, self.after_bmp)

    def local_hist_eq(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        gray:MatLike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        result = local_his_eq(gray,500)
        self.proc_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.display_image(self.proc_img, self.after_bmp)

    def enhance(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape)==2 else img.copy()
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(img_bgr, -1, kernel)
        blurred = cv2.GaussianBlur(img_bgr, (9, 9), 0)
        combined = cv2.addWeighted(sharpened, 0.4, cv2.addWeighted(img_bgr, 1.5, blurred, -0.5, 0), 0.6, 0)
        hsv = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        final = cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)
        self.proc_img = final
        self.display_image(self.proc_img, self.after_bmp)

    def gray_enhance(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        self.proc_img = cv2.equalizeHist(gray)
        self.display_image(self.proc_img, self.after_bmp)

if __name__ == '__main__':
    # img = cv2.imread(r"C:\Users\mrmmo\Desktop\nnn.jpg", cv2.IMREAD_GRAYSCALE)
    # assert img is not None, "file could not be read, check with os.path.exists()"
    
    # hist,bins = np.histogram(img.flatten(),256,[0,256])
    
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * float(hist.max()) / cdf.max()
    
    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(x=img.flatten(),bins=256,range=[0,256], color = 'r')
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')
    # plt.show()
    # cdf_m = np.ma.masked_equal(cdf,0)
    # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # cdf = np.ma.filled(cdf_m,0).astype('uint8')
    # img2 = cdf[img]
    # cv2.imshow('img2', img2)
    
    app = wx.App()
    frame = ImageProcessingGUI(None, 'Image Processing GUI')
    frame.load_image(r"C:\Users\mrmmo\Desktop\nnn.jpg")
    app.MainLoop()