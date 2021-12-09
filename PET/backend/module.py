import json

key = './ocr/key_ocr.txt'
with open(key) as f: KEY = f.read()
# 본인의 Secret Key로 치환

URL = './ocr/url_ocr.txt'
with open(URL) as f: URL = f.read()
# 본인의 APIGW Invoke URL로 치환

# json 데이터 가져오기 
with open('./users/pet_num.json', 'r') as fp:
    data = json.load(fp)

classes = []
# class의 이름들
with open("./data/custom.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


# ocr로 사용자의 정보를 입력 받는다 
# Module 1 : 사진파일 >> 텍스트 
def call_picture(pic):
  import json
  import base64
  import requests

  f = open(pic, "rb")
  img = base64.b64encode(f.read())

  headers = {
      "Content-Type": "application/json",
      "X-OCR-SECRET": KEY
  }
      
  data = {
      "version": "V2",
      "requestId": "string",
      "timestamp": 0,         # 현재 시간값
      "lang":"ko",
      "images": [
          {
              "name": "sample_image",
              "format": "png",
              "data": img.decode('utf-8')
            # "templateIds": [400]  # 설정하지 않을 경우, 자동으로 템플릿을 찾음 
          }
      ]
  }
  data = json.dumps(data)
  response = requests.post(URL, data=data, headers=headers)
  res = json.loads(response.text)
  id = res['images'][0]['title']['inferText']
  return id


# Module 2 : YOLO 함수
def yolo(frame, size, score_threshold, nms_threshold):
    # Module 2 모델에 필요한 것 
    import cv2
    import numpy as np  

    # 이미지 resize
    frame = cv2.resize(frame, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)

    net = cv2.dnn.readNetFromDarknet("./data/yolov4.cfg", "./data/yolov4_last.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    layer_names = net.getLayerNames()

    # net.getUnconnectedOutLayers()의 리턴값이 [000, 000, 000] 이었음
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 이미지의 높이, 너비, 채널 받아오기
    height, width, channels = frame.shape

    # 네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (size, size), (0, 0, 0), True, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(blob)

    # 결과 받아오기
    outs = net.forward(output_layers)

    # object_data 객체 선언
    # 각 데이터들을 저장할 리스트 선언
    object_data = {
      'class_ids' : [],
      'confidences' : [],
      'boxes' : [],
      'indexes' : -1
    }

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                object_data['boxes'].append([x, y, w, h])
                object_data['confidences'].append(float(confidence))
                object_data['class_ids'].append(class_id)

    # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
    # print(f"boxes: {object_data['boxes']}")
    # print(f"confidences: {object_data['confidences']}")

    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
    indexes = cv2.dnn.NMSBoxes(object_data['boxes'], object_data['confidences'], score_threshold=score_threshold, nms_threshold=nms_threshold)
    object_data['indexes'] = indexes

    # 후보 박스 중 선택된 박스의 인덱스 출력
    # print(f"indexes: ", end='')
    # for index in indexes:
    #     print(index, end=' ')
    # print("\n\n============================== classes ==============================")

    for i in range(len(object_data['boxes'])):
        if i in indexes:
            x, y, w, h = object_data['boxes'][i]
            class_name = classes[object_data['class_ids'][i]]
            label = f"{class_name} {object_data['confidences'][i]:.2f}"
            color = colors[object_data['class_ids'][i]]

            # 사각형 테두리 그리기 및 텍스트 쓰기
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            
            # 탐지된 객체의 정보 출력
            print(f"[{class_name}({i})] conf: {object_data['confidences'][i]} / x: {x} / y: {y} / width: {w} / height: {h}")

    return frame, object_data

# Module 2 : 패트병 갯수 파악 
def pet_total(image_list):
    import cv2
    
    p_count = 0
    f_count = 0
    n_count = 0

    for img in image_list:
        image = img
        frame = cv2.imread(image)
        size = 256

        frame, object_data = yolo(frame=frame, size=size, score_threshold=0.4, nms_threshold=0.4)
        
        path = './static/images/'
        cv2.imwrite(path + 'pre_' + img.split('/')[-1] , frame)
        
        try:
            flag = object_data['class_ids'][object_data['indexes'][0]] # 이걸 판별해서
            #print(f'페트병을 분리함에 넣어주세요.') 
        except:
            #print(f'이미지 인식 실패.')
            n_count += 1
            continue

        if flag == 0:
            p_count += 1
        else: 
            f_count += 1

    print(f'총 {p_count + f_count + n_count}의 페트병 중 {p_count}개의 페트병이 통과되었고, {n_count}개의 페트병 이미지는 인식하지 못했습니다.')
    return p_count


# Module 3 : FAST_API 적용 
# 아이디, 페트병 개수, 리워드 현황 출력 
# json 파일로 변경사항 저장 
def pet_sum(user_id:str, pet_num:int):
  reward = 0
  if user_id in list(data.keys()):
    data[user_id][0]+= pet_num
    # 포인트 지급 
    data[user_id][1] = 10 * data[user_id][0]
    # 저장 
    with open('./users/pet_num.json', 'w') as fp:
      json.dump(data,fp)
    return data[user_id]
  else:
    new_m_p = {user_id : [pet_num, reward]}
    data.update(new_m_p)
    # 포인트 지급 
    data[user_id][1] = 10 * data[user_id][0]
    #저장 
    with open('./users/pet_num.json', 'w') as fp:
      json.dump(data,fp)
    return data[user_id]
