## 전처리 function

- 본 저장소에 있는 preprocess.py, utils.py, u2net.py를 모두 필요로 함

- 위 python file들과 동일 경로에 pretrained_models directory를 생성하고, 그 내부에 다음 pretrained model을 저장한다.

    - https://drive.google.com/file/d/1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P/view?usp=sharing

- 사용법

    ```python
    from preprocess import *
    
    # 업로드된 requests.files['file'] FileStorage 객체를 인자로 제공
    temp_image, masked_temp_image = human_binarization(requests.files['file'])
    # 임시로 resized된 이미지 및 binary mask 이미지를 우리가 사용할 target_size로 리사이즈
    resized_image, resized_mask = resizing(tmp_image, masked_tmp_image)
    ```

    

