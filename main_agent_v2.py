import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import queue
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------
# 1. 설정 (User Preferences)
# ---------------------------------------------------------
MY_API_KEY = "" 
MODEL_NAME = "gemini-3-flash-preview"

# [설정] 초기 자세 (Home Pose)
HOME_QPOS = np.array([0, 0, 0, -1.5708, 0, 1.5708, -0.7853, 0.04, 0.04])

# [설정] 부드러운 움직임 계수
SMOOTHING_FACTOR = 0.05 

SCENE_XML = "scene.xml"

# ---------------------------------------------------------
# 2. Gemini 통신 스레드 클래스
# ---------------------------------------------------------
class GeminiBrain(threading.Thread):
    def __init__(self, client, model_name, result_queue):
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.result_queue = result_queue
        self.input_image = None
        self.is_thinking = False
        self.daemon = True 

    def think(self, image):
        if self.is_thinking: return 
        self.input_image = image
        self.is_thinking = True
        self.start_processing()

    def start_processing(self):
        threading.Thread(target=self._api_call).start()

    def _api_call(self):
        try:
            prompt = """
            Look at the simulation screen. Find the Red Cube on the table.
            I need the 3D position [x, y, z] of the center of the Red Cube relative to the robot base.
            Coordinate System Assumption: Robot base is at [0, 0, 0].
            Output ONLY the python list format e.g., [0.5, 0.1, 0.025].
            Do not provide any explanation.
            """
            
            config = types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ]
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, self.input_image],
                config=config
            )
            
            if not response.text:
                print(f"\n⚠️ [Gemini Warning] 응답 내용 없음. 재시도.")
                return

            text = response.text.strip()
            start, end = text.find('['), text.find(']')
            
            if start != -1 and end != -1:
                coord_str = text[start:end+1]
                import ast
                target_pos = np.array(ast.literal_eval(coord_str))
                self.result_queue.put(target_pos)
                print(f"\n🧠 [Gemini Update] 목표 발견! {target_pos}")
            else:
                print(f"\n⚠️ [Gemini] 좌표 형식 아님: {text}")

        except Exception as e:
            print(f"\n❌ [Gemini Error] {e}")
        finally:
            self.is_thinking = False

# ---------------------------------------------------------
# 3. 수치적 IK 함수
# ---------------------------------------------------------
def solve_ik(model, data, target_pos, current_qpos):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    current_pos = data.xpos[body_id]
    error = target_pos - current_pos
    
    if np.linalg.norm(error) > 0.05:
        error = error / np.linalg.norm(error) * 0.05

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    J = jacp[:, :7]
    lambda_sq = 0.01
    dq = J.T @ np.linalg.solve(J @ J.T + lambda_sq * np.eye(3), error)
    return dq

# ---------------------------------------------------------
# 4. 메인 루프
# ---------------------------------------------------------
def main():
    client = genai.Client(api_key=MY_API_KEY)
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    # 초기화
    data.qpos[:9] = HOME_QPOS
    data.ctrl[:] = HOME_QPOS[:8]
    data.ctrl[7] = 255 
    mujoco.mj_forward(model, data)

    brain_queue = queue.Queue()
    brain = GeminiBrain(client, MODEL_NAME, brain_queue)
    
    current_target_pos = None
    target_qpos = HOME_QPOS[:7].copy()
    
    print("🦾 [System] 시뮬레이션 시작. (자동 스캔 중...)")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_think_time = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            # A. 뇌 업데이트
            if not brain_queue.empty():
                new_pos = brain_queue.get()
                current_target_pos = new_pos 
            
            now = time.time()
            if not brain.is_thinking and (now - last_think_time > 4.0):
                print("📸 [Scan] Gemini에게 화면 전송...")
                renderer.update_scene(data)
                pixels = renderer.render()
                img = Image.fromarray(pixels)
                brain.think(img)
                last_think_time = now

            # B. 동작 계획
            if current_target_pos is not None:
                dq = solve_ik(model, data, current_target_pos, data.qpos[:7])
                target_qpos += dq * 0.5
            else:
                target_qpos = HOME_QPOS[:7]

            # C. 부드러운 제어
            current_ctrl = data.ctrl[:7].copy()
            next_ctrl = current_ctrl * (1 - SMOOTHING_FACTOR) + target_qpos * SMOOTHING_FACTOR
            data.ctrl[:7] = next_ctrl
            
            # --- [수정된 부분] 거리 계산 로직 간소화 ---
            if current_target_pos is not None:
                # 존재하지 않는 'attachment_site' 대신, 확실히 존재하는 'hand' body 사용
                hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
                hand_pos = data.xpos[hand_id]
                
                dist = np.linalg.norm(hand_pos - current_target_pos)
                
                # 목표에 가까워지면(0.1m 이내) 그리퍼 닫기 시도 (예시)
                # 시각적으로 확인하기 위해, 가까이 가면 0(닫힘), 멀면 255(열림)
                if dist < 0.1:
                    data.ctrl[7] = 0.0 # 잡기!
                else:
                    data.ctrl[7] = 255.0 # 열기

            # D. 물리 스텝
            mujoco.mj_step(model, data)
            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()