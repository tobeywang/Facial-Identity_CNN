Facial-Identity_CNN
===
using pillow and openCV package to get Image then to run facial identity by cnn algorithm

## env install

Python version 3.5 up

## Python package install
`$ pip install PIL ` `$ pip install numpy `
`$ pip install glob ` `$ pip install cv2 ` 
`$ pip install logging ` 

資料夾架構
---
- img          訓練用的完整圖片
- img_face     預測圖片的所有小臉圖片
- img_face_plt 框出預測完整圖片的臉與預測結果
- test         訓練時測試段小臉圖片
- train        訓練時訓練段每一個人一個資料夾並放入對應小臉圖片(越多越好)
- valild       訓練時驗證段完整圖片

結果
---
![](https://i.imgur.com/gH9JFNC.jpg)


###### tags: `python` `opencv` `keras`
