# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 포즈(관절) 추출 모듈 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 웹캠 연결
cap = cv2.VideoCapture(0)

print("카메라 연동 완료. 테스트를 시작합니다. 종료하려면 'q'를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 영상을 불러올 수 없습니다.")
        break

    # MediaPipe 처리를 위해 BGR 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 영상에서 관절 좌표 추출
    results = pose.process(image_rgb)

    status = "NORMAL"
    color = (0, 255, 0) # 초록색

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 화면의 가로, 세로 크기 가져오기
        h, w, c = frame.shape
        
        # 1. 뼈대 그리기 (원본 영상을 서버로 보내지 않고 이 뼈대 데이터만 사용함을 시각화)
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )

        # 2. 주요 관절의 y좌표 추출 (y좌표는 맨 위가 0, 맨 아래가 1입니다)
        # 코(머리), 양쪽 어깨, 양쪽 골반의 좌표
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # 3. 간단한 낙상 판별 로직 적용
        # 로직 A: 상체 중심(어깨)과 하체 중심(골반)의 높이 차이 계산
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_hip_y = (left_hip.y + right_hip.y) / 2
        
        # 서 있을 때는 어깨가 골반보다 위에 있음(y값이 더 작음)
        # 만약 머리(코)가 골반보다 아래로 내려가거나(y값이 커지거나), 
        # 어깨와 골반의 높이가 거의 비슷해지면(누운 상태) 낙상으로 판단
        
        height_diff = abs(avg_shoulder_y - avg_hip_y)

        # 코(머리)가 골반보다 낮아지거나, 어깨-골반 수직 거리가 매우 짧아진 경우
        if nose.y > avg_hip_y or height_diff < 0.15:
            status = "FALL DETECTED!"
            color = (0, 0, 255) # 빨간색

    # 화면에 상태 텍스트 출력
    cv2.putText(frame, status, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5, cv2.LINE_AA)

    # 결과 화면 보여주기
    cv2.imshow('Fall Detection Test (Press "q" to quit)', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()