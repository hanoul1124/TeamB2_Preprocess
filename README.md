## 전처리 function

- 본 저장소에 있는 모든 py file을 필요로 한다

- 위 python file들과 동일 경로에 pretrained_models directory를 생성하고, 그 내부에 다음 pretrained model을 저장한다.

- ```bash
    main.py
    preprocess.py
    binarization.py
    u2net.py
    create_model.py
    resize.py
    utils.py
    # human_binarization에는 u2net_human_seg.pth, cloth_binarization에는 unet_cloth_seg.pth
    pretrained_models/u2net_human_seg.pth, unet_cloth_seg.pth
    ```

    - human_seg.pth : https://drive.google.com/file/d/1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P/view?usp=sharing

    - cloth_seg.pth: https://github.com/ternaus/cloths_segmentation/releases/download/0.0.1/weights.zip

- 사용법

    ```python
    # Human binarizatoin
    from preprocess import resizing_human
    
    # 업로드된 requests.files['file'] FileStorage 객체를 인자로 제공
    resized_image = resizing_human(<image_file>, <temp_size>)
    
    # temp_size : Unet Inference 전 소요시간을 줄이기 위한 임시 리사이징 크기
    # default = 512로, 꼭 필요한 파라미터는 아님
    resized_image = resizing_human(requests.files['file'])
    ```
    
    ```bash
    # Cloth binarization
    python main.py resize-cloth <Image path> <Save path> --temp-size <int:temp_size>
    ```
    
    

