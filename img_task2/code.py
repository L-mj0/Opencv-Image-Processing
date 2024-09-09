import random

import cv2
import os
import numpy as np
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import qrcode
import pyzbar.pyzbar as pyzbar

# 全局 -> 图片，缩放因子
img = None
width_factor = None
height_factor = None
resize_factor = None



# 上传图片
def upload_img():
    global img, width_factor, height_factor, resize_factor
    filename = filedialog.askopenfilename()

    if filename:
        img = cv2.imread(filename,-1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 将NumPy数组转换为PIL图像对象
        img_pil = Image.fromarray(img_rgb)
        showpicture.delete("all")
        # 获取缩放因子
        width_factor = showpicture.winfo_width() / img_pil.width
        height_factor = showpicture.winfo_height() / img_pil.height
        resize_factor = min(width_factor, height_factor)
        # 调整显示图像的大小 利用重采样过滤器，用于在图像缩放时进行高质量的重采样
        # 使用Lanczos重采样算法，这是一种插值方法，通常用于图像缩小，因为它能够很好地保留图像的细节和减少失真
        img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)), Image.LANCZOS)
        
        # 转换为Tkinter可用的图像对象
        img_tk = ImageTk.PhotoImage(img_resized)

        # 展示图片 获取当前canvas宽高的一半，将图片放在中心
        showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,image=img_tk)
        showpicture.image = img_tk

        # 显示文件名
        file_label.config(text=os.path.basename(filename))


# 向右旋转图片
def revolve_r_img():
    global img, width_factor, height_factor, resize_factor

    if img is not None:
        # 使用NumPy函数图像顺时针旋转90度
        img = np.rot90(img)

        # 调整图像映射到窗口上
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)  # 转换为 PIL 图像对象
        # print(width_factor,height_factor,resize_factor)
        img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                     Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)  # 将图像转换为 ImageTk.PhotoImage 对象

        showpicture.delete("all")
        showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                 image=img_tk)
        showpicture.image = img_tk


# 向左旋转图片
def revolve_l_img():
    global img, width_factor, height_factor, resize_factor

    if img is not None:
        # 使用NumPy函数将图像逆时针旋转90度
        img = np.rot90(img, k=-1)

        # 同上把图像映射到窗口上
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)  # 转换为 PIL 图像对象
        # print(width_factor,height_factor,resize_factor)
        img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                     Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)  # 将图像转换为 ImageTk.PhotoImage 对象

        showpicture.delete("all")
        showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                 image=img_tk)
        showpicture.image = img_tk



# 灰度化图片
def bw_img():
    global img
    if img is None:
        messagebox.showinfo("Error", "图片为空，请上传图片")
        return
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # 将彩色图像转换为灰度图像 直接映射到页面上
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 将灰度图像转换为PIL图像对象
        img_pil = Image.fromarray(img_gray)

        img_resize = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                    Image.LANCZOS)

        img = np.array(img_resize)

        img_tk = ImageTk.PhotoImage(img_resize)

        showpicture.delete("all")
        showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                 image=img_tk)
        showpicture.image = img_tk

    else:
        messagebox.showinfo("Error", "图片已经是灰度图像")
        return


# 根据坐标裁剪图片
'''
1. 判断是否在window中有图片img是否为空，如果为空，弹出提示：“图片为空，请上传图片”
2. 如果window内有图片，就弹出resize窗口
3. resize窗口中填写要截取的x1,y1,x2,y2坐标值，并加一个确认按钮
4. 点击确认按钮后，将裁剪的图片显示在window窗口上，并自动销毁resize窗口
'''


def resize():
    global img

    if img is None:
        messagebox.showinfo("Error", "图片为空，请上传图片")
        return

    def update_crop(event=None):
        x1 = int(x1_slider.get())
        y1 = int(y1_slider.get())
        x2 = int(x2_slider.get())
        y2 = int(y2_slider.get())

        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 >= x2 or y1 >= y2 or x2 > img.shape[1] or y2 > img.shape[0]:
            return

        cropped_img = img[y1:y2, x1:x2]
        img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                     Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        showpicture.delete("all")
        showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                 image=img_tk)
        showpicture.image = img_tk

    def apply_crop():
        global img
        x1 = int(x1_slider.get())
        y1 = int(y1_slider.get())
        x2 = int(x2_slider.get())
        y2 = int(y2_slider.get())

        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 >= x2 or y1 >= y2 or x2 > img.shape[1] or y2 > img.shape[0]:
            messagebox.showinfo("Error", f"输入的坐标不合法，请输入图片正确范围（{img.shape[0]}x{img.shape[1]})")
            return

        img = img[y1:y2, x1:x2]
        crop_window.destroy()

        update_main_window()

    crop_window = Toplevel(window)
    crop_window.title('Resize')
    crop_window.minsize(300, 200)

    x1_label = Label(crop_window, text="x1:")
    x1_label.grid(row=1, column=0)
    x1_slider = Scale(crop_window, from_=0, to=img.shape[1], orient=HORIZONTAL, command=update_crop)
    x1_slider.grid(row=1, column=1)

    y1_label = Label(crop_window, text="y1:")
    y1_label.grid(row=2, column=0)
    y1_slider = Scale(crop_window, from_=0, to=img.shape[0], orient=HORIZONTAL, command=update_crop)
    y1_slider.grid(row=2, column=1)

    x2_label = Label(crop_window, text="x2:")
    x2_label.grid(row=3, column=0)
    x2_slider = Scale(crop_window, from_=0, to=img.shape[1], orient=HORIZONTAL, command=update_crop)
    x2_slider.set(img.shape[1])
    x2_slider.grid(row=3, column=1)

    y2_label = Label(crop_window, text="y2:")
    y2_label.grid(row=4, column=0)
    y2_slider = Scale(crop_window, from_=0, to=img.shape[0], orient=HORIZONTAL, command=update_crop)
    y2_slider.set(img.shape[0])
    y2_slider.grid(row=4, column=1)

    confirm_btn = Button(crop_window, text="确认", command=apply_crop)
    confirm_btn.grid(row=5, column=0, columnspan=2)

    def update_main_window():
        global img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                     Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        showpicture.delete("all")
        showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                 image=img_tk)
        showpicture.image = img_tk

    crop_window.protocol("WM_DELETE_WINDOW", crop_window.destroy)


# 查看直方图
def img_histogram():
    global img
    if img is None:
        messagebox.showinfo("Error", "图片为空，请上传图片")
        return

    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)

        b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
        g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
        r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])

        plt.subplots(1, 3, figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.plot(r_hist, color='r')
        plt.title('Red Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.subplot(1, 3, 2)
        plt.plot(g_hist, color='green')
        plt.title('Green Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.subplot(1, 3, 3)
        plt.plot(b_hist, color='blue')
        plt.title('Blue Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    elif len(img.shape) == 2:
        # 灰度图像计算直方图
        hist_gray = cv2.calcHist([img], [0], None, [256], [0, 256])

        # 绘制直方图
        plt.plot(hist_gray, color='gray')
        plt.title('Grayscale Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()

    else:
        messagebox.showinfo("Error", "图像格式不支持")


# 保存图像
def save_img():
    global img
    if img is None:
        messagebox.showinfo("Error", "图片为空，请上传图片")
        return

    filename = filedialog.asksaveasfilename()
    if filename:
        cv2.imwrite(filename, img)


# 直方图均值化
def histogram_equalization():
    global img
    if img is None:
        messagebox.showinfo("Error", "图片为空，请上传图片")
        return
    elif len(img.shape) == 3 and img.shape[2] == 3:
        messagebox.showinfo("Error", "图片为彩色图像，请上传灰度图像")
        return
    else:
        img_grey = cv2.equalizeHist(img)
        # 映射到页面上
        img_pil = Image.fromarray(img_grey)
        width_factor = showpicture.winfo_width() / img_pil.width
        height_factor = showpicture.winfo_height() / img_pil.height
        resize_factor = min(width_factor, height_factor)
        img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                     Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        showpicture.delete("all")
        showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                 image=img_tk)
        showpicture.image = img_tk



# 调整图片的对比度
'''
1. 判断是否在window中有图片img是否为空，如果为空，弹出提示：“图片为空，请上传图片”
2. 如果window内有图片，就弹出contrast窗口;窗口中输入对比度参数 (alpha) 和亮度参数 (beta)
3. 点击确认按钮后，将调整后的图片显示在window窗口上，并自动销毁contrast窗口
'''

def contrast_img():
    global img, width_factor, height_factor, resize_factor
    if img is None:
        messagebox.showinfo("Error", "图片为空，请上传图片")
        return
    else:
        def update_contrast(event=None):
            global img

            a = alpha_slider.get()
            b = beta_slider.get()

            contrast_img = cv2.convertScaleAbs(img, alpha=a, beta=b)

            img_rgb = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            width_factor = showpicture.winfo_width() / img_pil.width
            height_factor = showpicture.winfo_height() / img_pil.height
            resize_factor = min(width_factor, height_factor)
            img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                         Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_resized)
            showpicture.delete("all")
            showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                     image=img_tk)
            showpicture.image = img_tk


        contrast_window = Toplevel(window)
        contrast_window.title('Contrast')
        contrast_window.minsize(300, 150)

        alpha_label = Label(contrast_window, text="Alpha:")
        alpha_label.grid(row=0, column=0)
        alpha_slider = Scale(contrast_window, from_=0.1, to=3.0, resolution=0.1, orient=HORIZONTAL, command=update_contrast)
        alpha_slider.set(1.0)  # Default value
        alpha_slider.grid(row=0, column=1)

        beta_label = Label(contrast_window, text="Beta:")
        beta_label.grid(row=1, column=0)
        beta_slider = Scale(contrast_window, from_=-100, to=100, orient=HORIZONTAL, command=update_contrast)
        beta_slider.set(0)  # Default value
        beta_slider.grid(row=1, column=1)

        update_contrast()  # Call the update function to initialize the display



# 高斯噪声
'''
1. 判断是否在window中有图片img是否为空及图片是否为灰度图像，如果为空，弹出提示：“图片为空，请上传图片” / 图片为彩色图像，请上传灰度图像
2. 弹出一个新的窗口，让用户输入高斯噪声的均值和标准差
3. 点击确认按钮后，生成高斯噪声并显示在window窗口上，并自动销毁高斯噪声窗口
'''
def gauss_noise():
    global img
    if img is None:
        messagebox.showinfo("Error", "图片为空，请上传图片")
        return
    elif len(img.shape) == 3 and img.shape[2] == 3:
        messagebox.showinfo("Error", "图片为彩色图像，请上传灰度图像")
        return
    else:
        def update_gaussian_noise(event=None):
            global img
            mean = mean_slider.get()
            sigma = sigma_slider.get()

            gauss = np.random.normal(mean, sigma, img.shape[:2])
            noisy_img = np.clip(img + gauss, a_min=0, a_max=255).astype(np.uint8)

            img_pil = Image.fromarray(noisy_img)
            width_factor = showpicture.winfo_width() / img_pil.width
            height_factor = showpicture.winfo_height() / img_pil.height
            resize_factor = min(width_factor, height_factor)
            img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                         Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_resized)

            showpicture.delete("all")
            showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                     image=img_tk)
            showpicture.image = img_tk

        noise_window = Toplevel(window)
        noise_window.title('Gaussian Noise')
        noise_window.minsize(300, 150)

        mean_label = Label(noise_window, text="Mean:")
        mean_label.grid(row=0, column=0)
        mean_slider = Scale(noise_window, from_=-100, to=100, orient=HORIZONTAL, command=update_gaussian_noise)
        mean_slider.grid(row=0, column=1)

        sigma_label = Label(noise_window, text="Sigma:")
        sigma_label.grid(row=1, column=0)
        sigma_slider = Scale(noise_window, from_=0, to=100, orient=HORIZONTAL, command=update_gaussian_noise)
        sigma_slider.grid(row=1, column=1)

        update_gaussian_noise()  # Call the update function to initialize the display



# sp加噪处理
def sp_noise():
    global img
    if img is None:
        messagebox.showinfo("Error", "图片为空，请上传图片")
        return
    elif len(img.shape) == 3 and img.shape[2] == 3:
        def update_sp_noise(event=None):
            global img
            salt = salt_slider.get() / 100.0
            pepper = pepper_slider.get() / 100.0

            h, w = img.shape[:2]
            img_sp = np.copy(img)

            # Add salt noise
            num_salt = np.ceil(salt * h * w)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
            img_sp[coords[0], coords[1], :] = 255

            # Add pepper noise
            num_pepper = np.ceil(pepper * h * w)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
            img_sp[coords[0], coords[1], :] = 0

            img_rgb = cv2.cvtColor(img_sp, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            width_factor = showpicture.winfo_width() / img_pil.width
            height_factor = showpicture.winfo_height() / img_pil.height
            resize_factor = min(width_factor, height_factor)
            img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                         Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_resized)

            showpicture.delete("all")
            showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                     image=img_tk)
            showpicture.image = img_tk

        noise_window = Toplevel(window)
        noise_window.title('Salt and Pepper Noise')

        salt_label = Label(noise_window, text="Salt:")
        salt_label.grid(row=0, column=0)
        salt_slider = Scale(noise_window, from_=0, to=50, orient=HORIZONTAL, command=update_sp_noise)
        salt_slider.grid(row=0, column=1)

        pepper_label = Label(noise_window, text="Pepper:")
        pepper_label.grid(row=1, column=0)
        pepper_slider = Scale(noise_window, from_=0, to=50, orient=HORIZONTAL, command=update_sp_noise)
        pepper_slider.grid(row=1, column=1)

        update_sp_noise()  # Call the update function to initialize the display
    else:
        messagebox.showinfo("Error", "图片为灰度图像，请上传彩色图像")
        return

def gauss():
    global img, width_factor, height_factor, resize_factor
    if img is None:
        messagebox.showinfo("Error", "图片为空，请上传图片")
        return

    def update_gaussian_blur(event=None):
        global img
        mean = mean_slider.get()
        ksize = ksize_slider.get()
        if ksize % 2 == 0:
            ksize += 1  # Ensure ksize is odd
        blurred_img = cv2.GaussianBlur(img, (ksize, ksize), mean)

        img_rgb = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                     Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        showpicture.delete("all")
        showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                 image=img_tk)
        showpicture.image = img_tk

    gauss_window = Toplevel(window)
    gauss_window.title('Gaussian Blur')
    gauss_window.minsize(200, 100)

    mean_label = Label(gauss_window, text="Mean:")
    mean_label.grid(row=0, column=0)
    mean_slider = Scale(gauss_window, from_=0, to=10, orient=HORIZONTAL, command=update_gaussian_blur)
    mean_slider.grid(row=0, column=1)

    ksize_label = Label(gauss_window, text="Kernel Size:")
    ksize_label.grid(row=1, column=0)
    ksize_slider = Scale(gauss_window, from_=1, to=50, orient=HORIZONTAL, command=update_gaussian_blur)
    ksize_slider.grid(row=1, column=1)

    update_gaussian_blur()  # Call the update function to initialize the display



def getqrcode():
    # 生成二维码
    def create_qr():
        # 获取需要加密的内容
        data = qr_entry.get()
        # 创建二维码对象
        qr = qrcode.QRCode(
            version=1, # 二维码版本 1 到 40，数字越大，二维码包含的数据量越大，二维码的尺寸也越大
            error_correction=qrcode.constants.ERROR_CORRECT_L, # 最低级别的错误纠正能
            box_size=10, # 控制二维码中每个“盒子”（模块）的像素大小
            border=4, # 边框
        )
        qr.add_data(data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill='black', back_color='white')
        qr_img.show()
        qr_window.destroy()
    
    print("qrcode")
    qr_window = Toplevel(window)
    qr_window.title('生成二维码')
    qr_window.minsize(300, 200)

    qr_label = Label(qr_window, text="输入二维码内容:")
    qr_label.grid(row=1, column=0)
    qr_entry = Entry(qr_window)
    qr_entry.grid(row=1, column=1)

    confirm_btn = Button(qr_window, text="生成", command=create_qr)
    confirm_btn.grid(row=2, column=0, columnspan=2)

# 识别二维码
def reqr_code():
    global img
    if img is None:
        messagebox.showinfo("Error", "图片为空，请上传图片")
        return 
    
    detector = cv2.QRCodeDetector()
    # 获取识别到的数据、顶点坐标数组和二值化的二维码图像
    data,vertices_array, binary_qrcode = detector.detectAndDecode(img)
    print(vertices_array)
    print(data)
    # 检测图像中的二维码，并遍历每个检测到的二维码 并框出
    QRCodes = pyzbar.decode(img)
    # print('QRCodes:',QRCodes)
    for QRCode in QRCodes:
        # print(QRCode)
        # 解码二维码的数据并转换为字符串形式
        stringData = QRCode.data.decode('utf-8')
        print("二维码字符串是：\"" + stringData + "\"")
        # 二维码在图像上的顶点坐标数组 将二维码的边界点坐标转换为 numpy 数组，绘制二维码的边框
        points = np.array([QRCode.polygon], np.int32)
        # points = np.array([vertices_array], np.int32)
        # numpy reshape() 
        points = points.reshape((-1,1,2))
        print(points)
        cv2.polylines(img, [points], True, (0,255,0), 5)
        rectPoints = QRCode.rect
        print(rectPoints)
        cv2.putText(img, stringData, (rectPoints[0], rectPoints[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    if QRCodes == []:
        points = np.array([vertices_array], np.int32)
        # points = np.array([vertices_array], np.int32)
        # numpy reshape() 
        points = points.reshape((-1,1,2))
        print(points)
        cv2.polylines(img, [points], True, (0,255,0), 1)
        cv2.putText(img, data, (vertices_array[0][0][0], vertices_array[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


    # 显示图像映射到页面上
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    width_factor = showpicture.winfo_width() / img_pil.width
    height_factor = showpicture.winfo_height() / img_pil.height
    resize_factor = min(width_factor, height_factor)
    img_resized = img_pil.resize((int(img_pil.width * resize_factor), int(img_pil.height * resize_factor)),
                                    Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_resized)

    img_tk = ImageTk.PhotoImage(img_resized)
    showpicture.delete("all")
    showpicture.create_image(showpicture.winfo_width() / 2, showpicture.winfo_height() / 2, anchor=CENTER,
                                image=img_tk)
    showpicture.image = img_tk

    if vertices_array is not None:
        messagebox.showinfo("QRCode", data)
    # else:
    #     messagebox.showinfo("Error", "无法识别二维码")



window = Tk()
window.title('Image pro')
window.minsize(640, 600)
window.configure(bg='black')
window.attributes('-alpha', 0.9)

# 创建菜单栏
menubar = Menu(window)
window.config(menu=menubar)

# 图像基本处理菜单
file_menu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Edit", menu=file_menu)
file_menu.add_command(label="Load image", command=upload_img)
file_menu.add_command(label="Resize", command=resize)
file_menu.add_command(label="Save", command=save_img)
file_menu.add_command(label="Exit", command=window.quit)

# 灰度化菜单
gray_menu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="灰度变换", menu=gray_menu)
gray_menu.add_command(label="灰度化", command=bw_img)
gray_menu.add_command(label="查看图片直方图", command=img_histogram)
gray_menu.add_command(label="直方图均衡（仅灰度图像）", command=histogram_equalization)
gray_menu.add_command(label="对比度增强", command=contrast_img)

# 噪声菜单
noise_menu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="噪声与滤波", menu=noise_menu)
noise_menu.add_command(label="高斯噪声", command=gauss_noise)
noise_menu.add_command(label="sp_噪声", command=sp_noise)
noise_menu.add_command(label="高斯模糊", command=gauss)

# 二维码菜单
code_menu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="二维码", menu=code_menu)
code_menu.add_command(label="二维码解码", command=reqr_code)
code_menu.add_command(label="生成二维码", command=getqrcode)



# 创建工具栏
topbar = Canvas(window, bg='black', width=640, height=50, highlightthickness=0, highlightbackground='black')
topbar.grid(row=0, column=0, columnspan=2)

img_upload = ImageTk.PhotoImage(Image.open("imgs/upload.png"))
upload_btn = Button(topbar, image=img_upload, command=upload_img, bd=1, bg='black', highlightthickness=0)
upload_btn.place(x=0, y=0)

img_revolver = ImageTk.PhotoImage(Image.open("imgs/revolve_r.png"))
revolver_btn = Button(topbar, image=img_revolver, command=revolve_r_img, bd=1, bg='black', highlightthickness=0)
revolver_btn.place(x=200, y=0)

img_revolvel = ImageTk.PhotoImage(Image.open("imgs/revolve_l.png"))
revolvel_btn = Button(topbar, image=img_revolvel, command=revolve_l_img, bd=1, bg='black', highlightthickness=0)
revolvel_btn.place(x=260, y=0)

img_bw = ImageTk.PhotoImage(Image.open("imgs/bw.png"))
bw_btn = Button(topbar, image=img_bw, command=bw_img, bd=1, bg='grey', highlightthickness=0)
bw_btn.place(x=320, y=0)

# 文件名
file_label = Label(topbar, text="", bg='black', fg='white', width=14)
file_label.place(x=68, y=10)

blank_frame = Frame(window, bg='black', height=20)
blank_frame.grid(row=1, column=0, columnspan=2)

showpicture = Canvas(window, bg='black', width=560, height=500, highlightthickness=0)
showpicture.grid(row=2, column=0, columnspan=2)

window.mainloop()
