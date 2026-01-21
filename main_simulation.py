import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
from google import genai
from PIL import Image
import io

# ---------------------------------------------------------
# 1. 설정 (API 키는 꼭 새로 발급받은 걸로 넣으세요!)
# ---------------------------------------------------------
MY_API_KEY = "AIzaSyBpBo1uoFaiwht8jy5VmwVxpCf11aE3bzg" 
client = genai.Client(api_key=MY_API_KEY)
MODEL_NAME = "gemini-3-flash-preview" # 혹은 "gemini-2.0-flash-exp"

# ---------------------------------------------------------
# 2. MuJoCo 환경 정의 (간단한 XML)
#    - 바닥(checkerboard), 조명, 빨간색 큐브 하나
# ---------------------------------------------------------
xml_string = """
<mujoco>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" />
    <geom name="floor" type="plane" size="1 1 0.1" rgba=".9 .9 .9 1" />
    <body name="box" pos="0 0 0.1">
      <joint type="free" />
      <geom type="box" size=".05 .05 .05" rgba="1 0 0 1" />
    </body>
  </worldbody>
</mujoco>
"""

# ---------------------------------------------------------
# 3. 메인 실행 루프
# ---------------------------------------------------------
def main():
    # 모델 로드
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    # 오프스크린 렌더링을 위한 설정 (눈 만들기)
    renderer = mujoco.Renderer(model, height=480, width=640)

    print("🦾 시뮬레이션 시작... (Space: 캡처 및 Gemini 질문, ESC: 종료)")

    # 뷰어 실행
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 물리 연산 (1 step)
            mujoco.mj_step(model, data)
            
            # 뷰어 업데이트
            viewer.sync()

            # --- [핵심] 키보드 입력 처리 ---
            # 주의: Passive viewer는 키 입력을 직접 받기 까다로울 수 있어
            # 터미널에서 Enter를 치거나 하는 방식으로 트리거할 수도 있지만,
            # 여기서는 주기적으로(예: 5초마다) 캡처한다고 가정하거나
            # 간단히 루프 도는 걸 먼저 보여드립니다.
            
            # (테스트용) 처음 1초 뒤에 딱 한 번만 Gemini에게 물어보기
            if 1.0 < data.time < 1.02: 
                print("\n📸 [찰칵] 화면 캡처 중...")
                
                # 1. 렌더러 업데이트
                renderer.update_scene(data)
                
                # 2. 이미지 배열 가져오기 (RGB)
                pixels = renderer.render()
                
                # 3. PIL 이미지로 변환
                img = Image.fromarray(pixels)
                
                # 4. Gemini에게 전송
                print("📤 Gemini에게 이미지 전송 중...")
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=["이 화면에 무엇이 보이니? 로봇 공학 관점에서 설명해줘.", img]
                )
                print(f"🧠 [Gemini 분석]:\n{response.text}")
                
                # 중복 전송 방지를 위해 시간 딜레이
                time.sleep(1)

            # 프레임 속도 조절
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()