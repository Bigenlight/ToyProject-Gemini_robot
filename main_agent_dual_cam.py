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

# [New] 사용할 카메라 2개 이름 (XML에 정의된 이름)
CAMERAS = ["front_cam", "side_cam"] 

try:
    key_path = os.path.join(os.path.dirname(os.getcwd()), "api_key.txt")
    with open(key_path, "r") as f:
        MY_API_KEY = f.readline().strip()
    print(f"🔑 API Key 로드 성공")
except FileNotFoundError:
    print(f"❌ [Error] API 키 파일을 찾을 수 없습니다.")
    sys.exit(1)

# 초기 자세
HOME_QPOS = np.array([0, 0, 0, -1.5708, 0, 1.5708, 0.732, 0.04, 0.04])
HOME_EE_POS = np.array([0.555, 0.0, 0.524])

# 속도 제한
MAX_JOINT_VELOCITY = 0.05 
FIXED_ROTATION = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

# ---------------------------------------------------------
# 2. Gemini 두뇌 (듀얼 카메라 버전)
# ---------------------------------------------------------
class GeminiBrain(threading.Thread):
    def __init__(self, client, model_name, result_queue):
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.result_queue = result_queue
        self.input_images = [] # 이미지를 리스트로 저장
        self.current_ee_pos = None 
        self.is_thinking = False
        self.daemon = True 

    # [수정] images 리스트를 받도록 변경
    def think(self, images, current_pos):
        if self.is_thinking: return 
        self.input_images = images # [front_img, side_img]
        self.current_ee_pos = current_pos
        self.is_thinking = True
        self.start_processing()

    def start_processing(self):
        threading.Thread(target=self._api_call).start()

    def _api_call(self):
        start_time = time.time()
        try:
            curr_str = f"[{self.current_ee_pos[0]:.2f}, {self.current_ee_pos[1]:.2f}, {self.current_ee_pos[2]:.2f}]" if self.current_ee_pos is not None else "Unknown"

            # [프롬프트 수정] 2개의 뷰를 활용하라고 명시
            prompt = f"""
            You are controlling a robot arm in MuJoCo simulation. You are provided with TWO images from different angles.
            - Image 1: Front View
            - Image 2: Side View
            
            Task:
            1. Identify the 'Green Sphere' (End-Effector) and 'Red Cube' (Target) in BOTH images.
            2. The 'Green Sphere' is currently at {curr_str}. Use this as a reference point.
            3. Triangulate the 3D position [x, y, z] of the Red Cube center relative to the robot base.
            
            [Visual Scale Info]
            - The floor has a grid pattern.
            - Blue checkerboard squares are 0.2m x 0.2m.
            - White line checkerboard squares (composed of 4 blue ones) are 0.4m x 0.4m.
            - The blue cube for helping seing the scale is located on [0.6, 0.4, 0.02].
            - The white cube for helping seing the scale is located on [0.2, 0.0, 0.02].
            - Floor height is z=0.0.
            
            [Constraint]
            - Output ONLY the python list format e.g., [0.5, 0.1, 0.025].
            """
            
            config = types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ]
            )

            # [핵심] contents 리스트에 텍스트와 이미지 2장을 순서대로 넣음
            # 순서: [프롬프트, "Front View:", 이미지1, "Side View:", 이미지2]
            content_payload = [
                prompt,
                "Image 1 (Front View):",
                self.input_images[0], 
                "Image 2 (Side View):",
                self.input_images[1]
            ]

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_payload,
                config=config
            )
            
            if not response.text: return

            text = response.text.strip()
            start, end = text.find('['), text.find(']')
            
            if start != -1 and end != -1:
                coord_str = text[start:end+1]
                import ast
                target_pos = np.array(ast.literal_eval(coord_str))
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

    error_pos = target_pos - current_pos
    error_rot = get_orientation_error(current_mat, target_rot)
    error_full = np.hstack([error_pos, error_rot])
    
    if np.linalg.norm(error_full) > 0.05:
        error_full = error_full / np.linalg.norm(error_full) * 0.05

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :7], jacr[:, :7]]) 
    
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
    
    current_target_pos = HOME_EE_POS.copy()
    is_tracking = False 
    
    print("🦾 [System] 듀얼 카메라 에이전트 시작. (초기 위치 대기 중)")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_think_time = 0
        last_print_time = 0 
        
        while viewer.is_running():
            step_start = time.time()
            now = time.time()

            # 1. 모니터링
            if now - last_print_time > 1.0:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME)
                curr = data.site_xpos[site_id]
                status = "Tracking" if is_tracking else "Waiting"
                print(f"📍 [{status}] EE Pos: [{curr[0]:.3f}, {curr[1]:.3f}, {curr[2]:.3f}]")
                last_print_time = now

            # 2. 뇌 업데이트
            if not brain_queue.empty():
                new_pos, duration = brain_queue.get()
                current_target_pos = new_pos 
                is_tracking = True 
                print(f"🚀 [Gemini] 타겟 수신(Dual View): {new_pos} (소요시간: {duration:.2f}s)")
            
            # 3. 뇌 요청 (듀얼 캡처)
            if not brain.is_thinking and (now - last_think_time > 4.0):
                print("📸 [Scan] 듀얼 카메라 촬영 중...")
                
                try:
                    captured_images = []
                    # [핵심] 두 카메라를 순회하며 촬영
                    for cam_name in CAMERAS:
                        renderer.update_scene(data, camera=cam_name)
                        pixels = renderer.render()
                        img = Image.fromarray(pixels)
                        captured_images.append(img)
                    
                    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME)
                    curr_pos_copy = data.site_xpos[site_id].copy()
                    
                    # 리스트로 이미지 전달
                    brain.think(captured_images, curr_pos_copy)
                    last_think_time = now
                    
                except Exception as e:
                    print(f"❌ 카메라 캡처 오류: {e}")
                    last_think_time = now + 10 

            # 4. 동작 제어
            dq = solve_ik(model, data, current_target_pos, FIXED_ROTATION, TARGET_SITE_NAME)
            dq = np.clip(dq, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)
            next_qpos = data.qpos[:7] + dq 
            data.ctrl[:7] = next_qpos

            # 5. 그리퍼 제어
            if is_tracking: 
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME)
                dist = np.linalg.norm(data.site_xpos[site_id] - current_target_pos)
                if dist < 0.05: data.ctrl[7] = 0.0 
                else: data.ctrl[7] = 255.0
            else:
                data.ctrl[7] = 255.0 

            # 6. 물리 스텝
            mujoco.mj_step(model, data)
            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()