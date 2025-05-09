import wx
import cv2
import numpy as np

# --- Image Processing Functions ---

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def edge_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def blur(img):
    return cv2.GaussianBlur(img, (15, 15), 0)

def hist_eq(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

# --- GUI Class ---
class ImageProcessingGUI(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(1000, 800),
                         style=wx.DEFAULT_FRAME_STYLE | wx.RESIZE_BORDER)
        self.original_img = None
        self.proc_img = None
        self.setup_ui()
        self.Show()

    def setup_ui(self):
        panel = wx.Panel(self)
        self.panel = panel
        panel.SetBackgroundColour(wx.Colour(32, 32, 48))  # Dark background color

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Load Image Button
        load_btn = wx.Button(panel, label="Load Image")
        load_btn.Bind(wx.EVT_BUTTON, self.load_image)
        sizer.Add(load_btn, 0, wx.ALL | wx.CENTER, 10)

        # Image Display Section
        disp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.before_label = wx.StaticText(panel, label="Before")
        self.after_label = wx.StaticText(panel, label="After")

        # Placeholder for before and after images
        self.before_bmp = wx.StaticBitmap(panel, size=(340, 340))
        self.after_bmp = wx.StaticBitmap(panel, size=(340, 340))

        # Set placeholders
        self.set_placeholder(self.before_bmp)
        self.set_placeholder(self.after_bmp)

        for label, bmp in [(self.before_label, self.before_bmp), (self.after_label, self.after_bmp)]:
            box = wx.BoxSizer(wx.VERTICAL)
            label.SetForegroundColour(wx.Colour(255, 255, 255))  # White text
            label.SetFont(wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            box.Add(label, 0, wx.CENTER | wx.BOTTOM, 5)
            box.Add(bmp, 0, wx.CENTER | wx.ALL, 5)
            disp_sizer.Add(box, 0, wx.ALL, 10)

        sizer.Add(disp_sizer, 0, wx.CENTER)

        # Filter Buttons
        btns = [
            ("Sharpen", self.sharpen),
            ("Edge Detection", self.edge_detect),
            ("Blur", self.blur),
            ("Histogram Equalization", self.hist_eq),
        ]

        grid_sizer = wx.GridSizer(rows=2, cols=2, vgap=10, hgap=10)
        self.buttons = []
        for label, handler in btns:
            btn = wx.Button(panel, label=label)
            btn.Bind(wx.EVT_BUTTON, handler)
            self.buttons.append(btn)
            grid_sizer.Add(btn, 0, wx.EXPAND)

        sizer.Add(grid_sizer, 0, wx.ALL | wx.CENTER, 10)

        # Back Button
        self.back_btn = wx.Button(panel, label="Back to Original")
        self.back_btn.Bind(wx.EVT_BUTTON, self.back_to_original)
        sizer.Add(self.back_btn, 0, wx.ALL | wx.CENTER, 10)

        panel.SetSizer(sizer)

    def set_placeholder(self, bmp_widget):
        placeholder = np.full((340, 340, 3), (240, 240, 240), dtype=np.uint8)  # Light gray
        bmp = wx.Bitmap.FromBuffer(340, 340, placeholder)
        bmp_widget.SetBitmap(bmp)

    def display_image(self, img, bmp_widget):
        if img is None:
            return
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        h, w = img_bgr.shape[:2]
        max_dim = 340
        aspect_ratio = w / h
        new_w, new_h = (max_dim, int(max_dim / aspect_ratio)) if aspect_ratio > 1 else (int(max_dim * aspect_ratio), max_dim)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((340, 340, 3), 240, dtype=np.uint8)  # Light gray background
        y_off = (340 - new_h) // 2
        x_off = (340 - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        bmp = wx.Bitmap.FromBuffer(340, 340, rgb)
        bmp_widget.SetBitmap(bmp)
        self.panel.Layout()

    def load_image(self, event=None):
        with wx.FileDialog(self, "Open Image", "", "",
                           "Image Files (*.png;*.jpg;*.jpeg)|*.png;*.jpg;*.jpeg",
                           wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                img = cv2.imread(path)
                if img is not None:
                    self.original_img = img
                    self.proc_img = None
                    self.display_image(self.original_img, self.before_bmp)
                    self.display_image(self.original_img, self.after_bmp)
                else:
                    wx.MessageBox("Failed to load image.", "Error")

    def get_current_img(self):
        return self.proc_img if self.proc_img is not None else self.original_img

    def back_to_original(self, event):
        if self.original_img is not None:
            self.proc_img = None
            self.display_image(self.original_img, self.after_bmp)

    def sharpen(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        self.proc_img = sharpen(img)
        self.display_image(self.proc_img, self.after_bmp)

    def edge_detect(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        self.proc_img = edge_detect(img)
        self.display_image(self.proc_img, self.after_bmp)

    def blur(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        self.proc_img = blur(img)
        self.display_image(self.proc_img, self.after_bmp)

    def hist_eq(self, event):
        img = self.get_current_img()
        if img is None:
            wx.MessageBox("Load an image first.", "Error")
            return
        self.proc_img = hist_eq(img)
        self.display_image(self.proc_img, self.after_bmp)

# --- Run App ---
if __name__ == '__main__':
    app = wx.App()
    frame = ImageProcessingGUI(None, 'Image Processing GUI - Simple Filters')
    app.MainLoop()
