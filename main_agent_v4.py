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
# 1. 설정 및 API 키
# ---------------------------------------------------------
MODEL_NAME = "gemini-3-flash-preview"
SCENE_XML = "scene.xml"
TARGET_SITE_NAME = "gripper" 

try:
    key_path = os.path.join(os.path.dirname(os.getcwd()), "api_key.txt")
    with open(key_path, "r") as f:
        MY_API_KEY = f.readline().strip()
    print(f"🔑 API Key 로드 성공: {key_path}")
except FileNotFoundError:
    print(f"❌ [Error] API 키 파일을 찾을 수 없습니다.")
    sys.exit(1)

# 초기 자세
HOME_QPOS = np.array([0, 0, 0, -1.5708, 0, 1.5708, 0.732, 0.04, 0.04])

# [New] 속도 제한 (Rad/step) - 로봇이 미쳐 날뛰는 것 방지
MAX_JOINT_VELOCITY = 0.05 

# 바닥 보기 회전 행렬
FIXED_ROTATION = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

# ---------------------------------------------------------
# 2. Gemini 두뇌 (모니터링 & 정보 강화)
# ---------------------------------------------------------
class GeminiBrain(threading.Thread):
    def __init__(self, client, model_name, result_queue):
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.result_queue = result_queue
        self.input_image = None
        self.current_ee_pos = None # 현재 손 위치 기억
        self.is_thinking = False
        self.daemon = True 

    def think(self, image, current_pos):
        if self.is_thinking: return 
        self.input_image = image
        self.current_ee_pos = current_pos
        self.is_thinking = True
        self.start_processing()

    def start_processing(self):
        threading.Thread(target=self._api_call).start()

    def _api_call(self):
        start_time = time.time() # 시간 측정 시작
        try:
            # 현재 위치 포맷팅
            curr_str = f"[{self.current_ee_pos[0]:.2f}, {self.current_ee_pos[1]:.2f}, {self.current_ee_pos[2]:.2f}]" if self.current_ee_pos is not None else "Unknown"

            # [Updated Prompt] 그리드 정보 및 현재 위치 추가
            prompt = f"""
            Look at the simulation screen.
            1. Identify the 'Green Sphere' (Current End-Effector) and the 'Red Cube' (Target).
            2. The 'Green Sphere' is currently at {curr_str}.
            3. Estimate the 3D position [x, y, z] of the Red Cube center relative to the robot base.
            
            [Visual Scale Info]
            - The floor has a grid pattern.
            - Blue squares are 0.2m x 0.2m.
            - White squares (composed of 4 blue ones) are 0.4m x 0.4m.
            
            [Constraint]
            - Robot base is at [0, 0, 0].
            - Table/Floor height is around z=0.0.
            
            Output ONLY the python list format e.g., [0.5, 0.1, 0.025].
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
                return

            text = response.text.strip()
            start, end = text.find('['), text.find(']')
            
            if start != -1 and end != -1:
                coord_str = text[start:end+1]
                import ast
                target_pos = np.array(ast.literal_eval(coord_str))
                
                # 결과 큐에 넣기 (좌표, 소요시간)
                duration = time.time() - start_time
                self.result_queue.put((target_pos, duration))
                
            else:
                print(f"\n⚠️ [Gemini] 좌표 파싱 실패: {text}")

        except Exception as e:
            print(f"\n❌ [Gemini Error] {e}")
        finally:
            self.is_thinking = False

# ---------------------------------------------------------
# 3. 6-DoF IK 함수
# ---------------------------------------------------------
def get_orientation_error(current_mat, target_mat):
    r_err_mat = target_mat @ current_mat.T
    quat_err = np.zeros(4)
    mujoco.mju_mat2Quat(quat_err, r_err_mat.flatten())
    if quat_err[0] < 0: quat_err = -quat_err
    rot_err = quat_err[1:] * 2.0
    return rot_err

def solve_ik(model, data, target_pos, target_rot, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    current_pos = data.site_xpos[site_id]
    current_mat = data.site_xmat[site_id].reshape(3, 3)

    # 에러 계산
    error_pos = target_pos - current_pos
    error_rot = get_orientation_error(current_mat, target_rot)
    error_full = np.hstack([error_pos, error_rot])
    
    # [안전장치 1] IK 타겟이 너무 멀면 에러 벡터를 잘라냄 (Clamping)
    # 한 번에 5cm 이상 계산하려 하지 마라.
    if np.linalg.norm(error_full) > 0.05:
        error_full = error_full / np.linalg.norm(error_full) * 0.05

    # Jacobian
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :7], jacr[:, :7]]) 
    
    # Solve
    diag = 0.05 * np.eye(6)
    dq = J.T @ np.linalg.solve(J @ J.T + diag, error_full)
    
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
    
    print("🦾 [System] 안정화 에이전트 시작.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_think_time = 0
        last_print_time = 0 # 모니터링 출력용
        
        while viewer.is_running():
            step_start = time.time()
            now = time.time()

            # ---------------------------------------
            # 1. 모니터링 (1초에 한 번 출력)
            # ---------------------------------------
            if now - last_print_time > 1.0:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME)
                curr = data.site_xpos[site_id]
                print(f"📍 [Robot] EE Pos: [{curr[0]:.3f}, {curr[1]:.3f}, {curr[2]:.3f}]")
                last_print_time = now

            # ---------------------------------------
            # 2. 뇌 업데이트 (결과 수신)
            # ---------------------------------------
            if not brain_queue.empty():
                new_pos, duration = brain_queue.get()
                current_target_pos = new_pos 
                print(f"🚀 [Gemini] 타겟 수신: {new_pos} (소요시간: {duration:.2f}s)")
            
            # ---------------------------------------
            # 3. 뇌 요청 (주기적)
            # ---------------------------------------
            if not brain.is_thinking and (now - last_think_time > 4.0):
                print("📸 [Scan] Gemini에게 요청 중...")
                renderer.update_scene(data)
                pixels = renderer.render()
                img = Image.fromarray(pixels)
                
                # 현재 로봇 손 위치도 같이 보냄 (프롬프트 힌트용)
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME)
                curr_pos_copy = data.site_xpos[site_id].copy()
                
                brain.think(img, curr_pos_copy)
                last_think_time = now

            # ---------------------------------------
            # 4. 동작 제어 (핵심 수정됨)
            # ---------------------------------------
            if current_target_pos is not None:
                # IK로 필요한 관절 속도(dq) 계산
                dq = solve_ik(model, data, current_target_pos, FIXED_ROTATION, TARGET_SITE_NAME)
                
                # [안전장치 2] 속도 제한 (Clamp Velocity)
                # 관절이 너무 빨리 돌지 않도록 자름
                dq = np.clip(dq, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)
                
                # [핵심 변경] 적분 누적(target_qpos += dq) 대신, 현재 상태 기반 업데이트
                # q_next = q_current + dq * gain
                # 이렇게 하면 로봇이 물리적으로 못 따라가도 목표값이 저 멀리 도망가지 않음
                next_qpos = data.qpos[:7] + dq 
                
                # 제어 입력
                data.ctrl[:7] = next_qpos
            else:
                # 타겟 없으면 제자리 유지
                data.ctrl[:7] = data.qpos[:7]

            # ---------------------------------------
            # 5. 그리퍼 제어
            # ---------------------------------------
            if current_target_pos is not None:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME)
                dist = np.linalg.norm(data.site_xpos[site_id] - current_target_pos)
                if dist < 0.05: data.ctrl[7] = 0.0 
                else: data.ctrl[7] = 255.0

            # 6. 물리 스텝
            mujoco.mj_step(model, data)
            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()