import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
from google import genai
from PIL import Image

# ---------------------------------------------------------
# 1. 설정 (API Key & Model)
# ---------------------------------------------------------
MY_API_KEY = "AIzaSyBpBo1uoFaiwht8jy5VmwVxpCf11aE3bzg" 
client = genai.Client(api_key=MY_API_KEY)
# ⚠️ 중요: 안정적인 1.5 Flash 모델 사용
MODEL_NAME = "gemini-3-flash-preview" # 혹은 "gemini-2.0-flash-exp"

SCENE_XML = "scene.xml"
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

# ---------------------------------------------------------
# 2. IK 함수 (아까 만든 것 재사용)
# ---------------------------------------------------------
def solve_ik(model, data, target_pos, body_name="hand"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    current_pos = data.xpos[body_id]
    error = target_pos - current_pos
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    J = jacp[:, :7]
    diag = 0.01 * np.eye(3)
    dq = J.T @ np.linalg.solve(J @ J.T + diag, error)
    return dq

# ---------------------------------------------------------
# 3. Gemini에게 좌표 물어보기 (Vision)
# ---------------------------------------------------------
def get_cube_position_from_gemini():
    # 1. 현재 화면 캡처
    renderer.update_scene(data)
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    
    print("📤 [Gemini] 화면 전송 중... (큐브 위치 찾는 중)")
    
    # 2. 프롬프트: 좌표만 딱 내놓으라고 강력하게 지시
    prompt = """
    Look at this simulation screen. There is a Red Cube on the table.
    I need the 3D position (x, y, z) of the Red Cube to move my robot arm.
    
    Estimation Rule:
    - The robot base is at (0,0,0).
    - The floor is at z=0.
    - The cube looks like it is around x=0.4 to 0.6.
    
    Output ONLY the python list format like [0.5, 0.0, 0.05]. 
    Do not say anything else. Just the numbers.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, img]
        )
        text = response.text.strip()
        print(f"🧠 [Gemini 응답]: {text}")
        
        # 3. 텍스트 -> 리스트 변환 (eval 사용은 조심해야 하지만 간단한 테스트엔 OK)
        # 예: "[0.5, 0.0, 0.05]" -> [0.5, 0.0, 0.05]
        import ast
        target_pos = ast.literal_eval(text)
        return np.array(target_pos)
        
    except Exception as e:
        print(f"❌ Gemini 인식 실패: {e}")
        return None

# ---------------------------------------------------------
# 4. 메인 루프
# ---------------------------------------------------------
def main():
    print("🦾 [System] 준비 완료! Space를 누르면 Gemini가 큐브를 찾습니다.")
    
    # 초기 자세
    home_qpos = model.keyframe('home').qpos.copy()
    data.qpos[:] = home_qpos
    mujoco.mj_forward(model, data)
    
    target_pos = None # 목표 위치 저장용

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # --- [입력 처리] Space 누르면 Gemini 호출 ---
            # (Viewer 창이 활성화된 상태에서 Space를 감지하려면 glfw 콜백이 필요하지만,
            #  여기서는 간단히 키보드 입력을 흉내내거나 주기적으로 실행하는 방식 대신
            #  '일정 시간마다' 혹은 '랜덤하게' 로직을 실행하는 게 쉽습니다.
            #  하지만 더 쉬운 방법: 그냥 시작 후 2초 뒤에 한 번 실행!)
            
            if 2.0 < data.time < 2.02 and target_pos is None:
                # 시뮬레이션 시간 2초 때 딱 한 번 실행
                detected_pos = get_cube_position_from_gemini()
                if detected_pos is not None:
                    target_pos = detected_pos
                    print(f"🎯 목표 설정 완료: {target_pos}로 이동합니다!")

            # --- [제어] 목표가 생기면 IK로 이동 ---
            if target_pos is not None:
                dq = solve_ik(model, data, target_pos)
                data.qpos[:7] += dq * 0.05 # 속도 조절
                data.ctrl[:7] = data.qpos[:7]
                data.ctrl[7] = 255 # 그리퍼 열기

            # 물리 연산
            mujoco.mj_step(model, data)
            viewer.sync()

            # 시간 동기화
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()