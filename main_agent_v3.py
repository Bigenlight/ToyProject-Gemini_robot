import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import queue
import os
import sys
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------
# 1. 설정 및 API 키 로드
# ---------------------------------------------------------
MODEL_NAME = "gemini-3-flash-preview"

# [설정] XML 파일 경로
SCENE_XML = "scene.xml"

# [설정] Site 이름 (XML에 추가한 Site의 정확한 이름!)
# 만약 틀리면 코드가 자동으로 알려줍니다.
TARGET_SITE_NAME = "gripper" 

# [API Key 로드] 상위 폴더의 api_key.txt 읽기
try:
    key_path = os.path.join(os.path.dirname(os.getcwd()), "api_key.txt")
    with open(key_path, "r") as f:
        MY_API_KEY = f.readline().strip()
    print(f"🔑 API Key 로드 성공: {key_path}")
except FileNotFoundError:
    print(f"❌ [Error] API 키 파일을 찾을 수 없습니다: {key_path}")
    print("코드를 실행하는 폴더의 상위 폴더에 'api_key.txt'가 있는지 확인해주세요.")
    sys.exit(1)

# [설정] 초기 자세 (Home Pose)
HOME_QPOS = np.array([0, 0, 0, -1.5708, 0, 1.5708, 0.732, 0.04, 0.04])

# [설정] 부드러운 움직임 계수
SMOOTHING_FACTOR = 0.05 

# [설정] 목표 회전 행렬 (그리퍼가 바닥을 향하는 자세)
FIXED_ROTATION = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

# ---------------------------------------------------------
# 2. Gemini 통신 스레드
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
# 3. 6-DoF IK 함수 (진단 기능 포함)
# ---------------------------------------------------------
def get_orientation_error(current_mat, target_mat):
    r_err_mat = target_mat @ current_mat.T
    quat_err = np.zeros(4)
    mujoco.mju_mat2Quat(quat_err, r_err_mat.flatten())
    if quat_err[0] < 0: quat_err = -quat_err
    rot_err = quat_err[1:] * 2.0
    return rot_err

def solve_ik(model, data, target_pos, target_rot, site_name):
    # Site ID 확인
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    except Exception:
        # 에러를 삼키지 않고 출력!
        print(f"❌ [IK Error] Site '{site_name}'를 찾을 수 없습니다!")
        return np.zeros(7)

    # 1. 위치 오차
    current_pos = data.site_xpos[site_id]
    error_pos = target_pos - current_pos
    
    # 2. 회전 오차
    current_mat = data.site_xmat[site_id].reshape(3, 3)
    error_rot = get_orientation_error(current_mat, target_rot)
    
    # 3. 전체 에러
    error_full = np.hstack([error_pos, error_rot])
    
    # 에러 클램핑
    if np.linalg.norm(error_full) > 0.1:
        error_full = error_full / np.linalg.norm(error_full) * 0.1

    # 4. Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    
    J_pos = jacp[:, :7]
    J_rot = jacr[:, :7]
    J_full = np.vstack([J_pos, J_rot]) 
    
    # 5. Solve
    diag = 0.05 * np.eye(6)
    dq = J_full.T @ np.linalg.solve(J_full @ J_full.T + diag, error_full)
    
    return dq

# ---------------------------------------------------------
# 4. 메인 루프
# ---------------------------------------------------------
def main():
    client = genai.Client(api_key=MY_API_KEY)
    
    # 모델 로드
    try:
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"❌ [XML Error] {SCENE_XML} 파일을 찾을 수 없습니다.")
        return

    renderer = mujoco.Renderer(model, height=480, width=640)

    # --- [Site 이름 진단] ---
    try:
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME)
        print(f"✅ Site '{TARGET_SITE_NAME}' 확인 완료.")
    except:
        print(f"\n🚨 [Critical Warning] Site '{TARGET_SITE_NAME}'가 XML에 없습니다!")
        print(f"   XML에 정의된 Site 목록을 확인하세요:")
        # Site 목록 출력 (ID 순회)
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            print(f"   - {name}")
        print("\n   → 위 목록 중 하나를 TARGET_SITE_NAME 변수에 넣어주세요.")
        return # 종료

    # 초기화
    data.qpos[:9] = HOME_QPOS
    data.ctrl[:] = HOME_QPOS[:8]
    data.ctrl[7] = 255 
    mujoco.mj_forward(model, data)

    brain_queue = queue.Queue()
    brain = GeminiBrain(client, MODEL_NAME, brain_queue)
    
    current_target_pos = None
    target_qpos = HOME_QPOS[:7].copy()
    
    print("🦾 [System] 에이전트 시작. (자동 스캔 중...)")
    
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
                # IK 계산
                dq = solve_ik(model, data, current_target_pos, FIXED_ROTATION, TARGET_SITE_NAME)
                
                # IK 결과가 0인지 체크 (디버깅용)
                if np.linalg.norm(dq) < 1e-6:
                   pass # 이미 목표에 도달했거나 계산 실패
                
                target_qpos += dq * 0.5 # Integration
            else:
                target_qpos = HOME_QPOS[:7]

            # C. 부드러운 제어
            current_ctrl = data.ctrl[:7].copy()
            next_ctrl = current_ctrl * (1 - SMOOTHING_FACTOR) + target_qpos * SMOOTHING_FACTOR
            data.ctrl[:7] = next_ctrl
            
            # D. 그리퍼 자동 제어
            if current_target_pos is not None:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME)
                current_site_pos = data.site_xpos[site_id]
                dist = np.linalg.norm(current_site_pos - current_target_pos)
                
                if dist < 0.05: # 5cm 이내
                    data.ctrl[7] = 0.0 # 잡기
                else:
                    data.ctrl[7] = 255.0 # 열기

            # E. 물리 스텝
            mujoco.mj_step(model, data)
            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()