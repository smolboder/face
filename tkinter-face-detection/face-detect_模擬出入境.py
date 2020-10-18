import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import csv
import datetime
from pyzbar.pyzbar import decode


class travelData():
    def __init__(self):
        self.encodeList = []  # 資料庫裡的人臉資料
        self.nameList = []  # 資料庫裡的人員資料姓名
        self.dateList = []  # 資料庫裡的人員資料 入境日期
        self.name = ''  # 初始預設旅客名字
        self.encode = np.array([])  # 初始預設 旅客臉部資訊
        self.recognition_result = False  # 半段是否辨識出人臉
        self.result_name = ''  # 從檔案庫裡的資料 核對辨識出的人臉姓名
        self.result_faceDis = 0.  # 從檔案庫裡的資料 核對辨識出的人臉相似的距離
        self.get_init_encodeList()  # 打開程式時 自動執行人臉資料讀取

    def get_init_encodeList(self):
        """讀取資料庫人員入境資料"""
        with open('人員入境資料.csv', 'r+', encoding='utf-8', newline='') as f:
            lines = csv.reader(f, delimiter=',')
            for line in list(lines)[1:]:
                faceEncode = np.array(eval(','.join(line[2:])))
                self.nameList.append(line[0])
                self.dateList.append(line[1])
                self.encodeList.append(faceEncode)


class ll(tk.Label):
    # 公有類別屬性
    # def __init__(self, master, *args, **kwargs):
    #     tk.Label.__init__(self, master=master, *args, **kwargs)  # 用 super()好像就沒有 要在哪個視窗master的問題

    def __init__(self, *args, **kwargs):  # 這樣也可以的樣子
        super().__init__(*args, **kwargs)  # 初始化

    def get_cap_image(self):
        """攝影機即時畫面更新攝影機畫面標籤的image"""
        global image2  # 沒加這個就沒辦法一直更新圖片
        ret, img = cap.read()
        img = img[:, ::-1, :]
        # print(img.shape)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 打開圖片
        img = Image.fromarray(img)
        image2 = ImageTk.PhotoImage(img)  # 變成tkinter可以使用的圖片
        label.configure(image=image2)  # 變更標籤顯示圖片的label內的值f

    def change_auto(self):
        """即時更新攝影機標籤內的畫面"""
        global image3
        self.get_cap_image()
        delay = 50  # 更換圖片間隔時間
        self.after(delay, self.change_auto)  # 400毫秒後進入遞迴

    def get_face_image(self):
        """擷取畫面並更新辨識畫面標籤的image"""
        global image3, travel, image4
        ret, img = cap.read()
        img = img[:, ::-1, :]
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 打開圖片
        img, returnData, faceOrcode = self.face_recognition(img)
        if faceOrcode == 'face' or faceOrcode == 'None':
            travel.encode = returnData
            img = Image.fromarray(img)
            image3 = ImageTk.PhotoImage(img)  # 變成tkinter可以使用的圖片
            label2.configure(image=image3)  # 變更標籤顯示圖片的label內的值f
        if faceOrcode == 'code':
            travel.name = returnData
            img = Image.fromarray(img)
            image4 = ImageTk.PhotoImage(img)  # 變成tkinter可以使用的圖片
            label3.configure(image=image4)  # 變更標籤顯示圖片的label內的值f
            labNone.configure(text='身分資料:\n\n' + travel.name)

    def face_recognition(self, image):
        """將得到的畫面進行人臉辨識並在圖上標示出 成功或者失敗 成功的話會加上臉部位置框框"""
        faceLoc = face_recognition.face_locations(image)
        encode = face_recognition.face_encodings(image)
        barcode = decode(image)
        if barcode:
            for bar in barcode:
                mydata = bar.data.decode('utf-8')
                x, y, w, h = bar.rect
                pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
                pts2 = np.float32([[0, 0], [170, 0], [0, 170], [170, 170]])
                M = cv.getPerspectiveTransform(pts1, pts2)
                image = cv.warpPerspective(image, M, (170, 170))
            return image, mydata, 'code'
        elif faceLoc:
            faceLoc = faceLoc[0]
            encode = encode[0]
            y1, x2, y2, x1 = faceLoc
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(image, (0, 430), (650, 468), (0, 255, 0), -1)
            cv.putText(image, f'{"success"}', (260, 456), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
            return image, encode, 'face'
        else:
            cv.rectangle(image, (0, 430), (650, 468), (255, 0, 0), -1)
            cv.putText(image, f'{"failed"}', (260, 456), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
            return image, encode, 'None'

    def save_face_data(self):
        """資料存檔"""
        global travel
        date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if travel.recognition_result and not travel.name in travel.nameList:
            with open('人員入境資料.csv', 'a', encoding='utf-8', newline='') as f:
                f.writelines(f'\n{travel.name},{date},{list(travel.encode)}')
            txt = travel.name + '\n\n' + date + '\n\n' + '存檔完成'
            labelText.configure(text=txt)
            travel.get_init_encodeList()  # 因為存進新資料所以必須更新encodeList資料 以便新資料的及時辨識
        else:
            txt = '資料庫已有此人資料\n\n無須存檔'
            labelText.configure(text=txt)

    def check_face(self):
        global travel
        try:
            faceDis = face_recognition.face_distance(travel.encodeList, travel.encode)
        except:
            txt = '請先拍照唷!!'
            travel.recognition_result = False
        else:
            travel.recognition_result = True
            index = np.argmin(faceDis)
            if not faceDis[index] < 0.5:
                txt = '資料庫查無此人資料~~'
            else:
                travel.result_name = travel.nameList[index]
                travel.result_faceDis = faceDis[index]
                txt = '辨識結果: ' + travel.result_name
                # txt = '辨識結果: ' + travel.result_name + '\n\n相似距離: ' + '{:.4f}'.format(travel.result_faceDis)
        labelText.configure(text=txt)


def init_Image(file, size):
    image = cv.imread(file)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, size, cv.INTER_AREA)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)  # tkinter 目前只吃png和gif 圖檔 所以要用這樣開啟jpg
    return image


# 建立 處理旅客資料 物件 用來儲存姓名以及臉部資料的物件 之後再存進資料庫
travel = travelData()
# tkinter部分
window = tk.Tk()
window.title('模擬機場入境')
window.geometry('1280x800')
# 主視窗裡建立攝影機畫面的子視窗
v = tk.Frame(window, width=1280, height=480, bg='black')  # 建立 子視窗
v.pack(side=tk.TOP, anchor=tk.S, expand=tk.YES, fill=tk.BOTH)
# 在攝影機畫面的子視窗裡的再建立兩個影像子視窗
a = tk.Frame(v, width=640, height=480, bg='light gray')  # 建立 子視窗
a.pack(side=tk.LEFT, anchor=tk.S, expand=tk.YES, fill=tk.BOTH)
b = tk.Frame(v, width=640, height=480, bg='light gray')  # 建立 子視窗
b.pack(side=tk.LEFT, anchor=tk.S, expand=tk.YES, fill=tk.BOTH)
# 主視窗裡建立按鈕以及文字標籤的子視窗
d = tk.Frame(window, width=300, height=200, bg='light gray')  # 建立 子視窗
d.pack(side=tk.TOP, anchor=tk.S, expand=tk.YES, fill=tk.BOTH)
# 攝影機畫面的子視窗裡的兩個子視窗 初始化所放置的圖片
# image = init_Image('../FACE_RECOGNITION/Test/tiai1.jpg',(640, 480))
image = init_Image('origin.png', (640, 480))
image3 = image
# 條碼辨識區塊的初始化圖片
image4 = init_Image('init.png', (170, 170))
# 開啟攝影機
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(10, 100)
# 顯示畫面 攝影機
L1 = tk.Label(a, text='攝影機畫面', font=10)
L1.pack(side=tk.TOP)
# 顯示畫面 辨識畫面
L2 = tk.Label(b, text='辨識畫面', font=10)
L2.pack(side=tk.TOP)
# 顯示 攝影機畫面的標籤
label = ll(a, image=image)  # 自創類別 繼承tk.Label  放圖片的label
label.pack(side=tk.TOP)
# 顯示 辨識畫面的標籤
label2 = ll(b, image=image3)  # 自創類別 繼承tk.Label  放圖片的label
label2.pack(side=tk.TOP)
# 顯示 辨識姓名 的顯示畫面
label3 = ll(d, image=image4)  # 自創類別 繼承tk.Label  放圖片的label
label3.grid(row=1, column=0, columnspan=1, rowspan=5, padx=70)
# 填滿位置用
labNone = tk.Label(d, text='你的名字', width=20, height=10, bg='light gray', font=15)
labNone.grid(row=1, column=2, columnspan=2, rowspan=5, padx=70)
# 身份辨識部分 左下角圖片及文字兩個標籤
labelText2 = tk.Label(d, text='身分條碼掃描', font=10)
labelText2.grid(row=0, column=0, padx=70)
labelText2 = tk.Label(d, text='辨識結果', font=10)
labelText2.grid(row=0, column=2, columnspan=2)
# 各個按扭
buta = tk.Button(d, text='開啟攝影機', command=label.change_auto, width=20)
buta.grid(row=1, column=4, padx=10)
buta = tk.Button(d, text='拍照', command=label.get_face_image, width=20)
buta.grid(row=2, column=4, padx=10)
buta = tk.Button(d, text='進行辨識', command=label.check_face, width=20)
buta.grid(row=3, column=4, padx=10)
buta = tk.Button(d, text='資料存檔', command=label.save_face_data, width=20)
buta.grid(row=4, column=4, padx=10)
# 文字輸出標籤
labelText = tk.Label(d, text='歡迎使用人臉辨識系統', width=47, height=10, font=15, bg='gray')
labelText.grid(row=1, column=5, columnspan=3, rowspan=5, padx=10)
# 結束程式按扭
tk.Button(d, text='結束程式', command=window.destroy, width=20).grid(row=5, column=4, padx=10)

window.mainloop()
