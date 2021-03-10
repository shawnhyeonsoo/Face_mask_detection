# Face_mask_detection
Detecting Face Masks using DNN implementations
</br>
To-dos: </br>
- Dataset 확보
- 모델 설계
- 성능 검증
</br>

From 102 Frontal Faces:
가상 마스크 씌웠을 때 인식률: 80/100

</br>
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

</br>
</br>
RMFD Dataset: </br>
</br>

CNN model : </br>
- c1: Convolution Layer: 32, (3 x 3), activation: ReLU
- c2: Convolution Layer: 32, (3 x 3), activation ReLU
- m3: Max Pooling: (2,2), Dropout (0.25)
- c4: Convolution Layer: 64, (3 x 3), activation: ReLU
- c5: Convolution Layer: 64, (3 x 3), activation: ReLU
- m6: Max Pooling, (2, 2) , Dropout (0.25)
- c6: Convolution Layer: 64, (3 x 3), activation: ReLU
- d7: Dropout (0.5)
- f8: Fully Connected Layers (64, activation: ReLU)
- f9: Fully Connected Layers (32, activation: ReLU)
- f10: Output Layer: (2, activation: Softmax)


######
2230 Masked Faces </br>
90468 Unmasked Faces </br>
Total = 2230 + 90468 = 
Training data: Validation Data = 0.85: 0.15 </br>
</br>
</br>
Accuracy Results from test on 2021. 03. 10: </br>
- On Validation set: 
- Masked Faces Only: 92.24%
- Unmasked Faces Only: 96.16% 


> Requires separate test data for clear comprehension
