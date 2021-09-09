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
    
    # case1
    # 업로드된 requests.files['file'] FileStorage 객체를 인자로 제공
    # resizing_human(<image_file>, <temp_size>, <model>)
    # temp_size : Unet Inference 전 소요시간을 줄이기 위한 임시 리사이징 크기. default = 512
    # model: 사전에 모델을 선언해 둔 경우. default = None
    resized_image = resizing_human(requests.files['file'])
    
    
    # case2
    # 사전에 서버에서 모델을 미리 생성해두고 싶은 경우, 다음을 통해 모델을 생성한다.
    from create_model import *
    from preprocess import resizing_human_with_model
    
    model = create_model('Unet_human')
    model.eval()
    
    # 모델을 생성하고 추론 모드로 설정한 이후, 다음 함수를 사용하여 동일한 동작을 수행
    resized_image = resizing_human(request.files['file'], model=model, temp_size=384)
    ```
    
    ```bash
    # Cloth binarization
    python main.py resize-cloth <Image path> <Save path> --temp-size <int:temp_size>
    ```
    
    

