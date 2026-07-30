import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import queue
import os
import sys
import json
import re
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------
# 1. 설정 및 API 키
# ---------------------------------------------------------
# MODEL_NAME = "gemini-2.0-flash-exp"  # 속도와 JSON 처리가 더 좋은 모델 권장 (없으면 기존 것 사용)
# MODEL_NAME = "gemini-3-flash-preview"
MODEL_NAME = "gemini-robotics-er-2-preview"  # Gemini Robotics-ER 2 (spatial reasoning 특화)
# MODEL_NAME = "gemini-1.5-flash" # 혹은 기존 사용하던 모델
SCENE_XML = "scene.xml"
TARGET_SITE_NAME = "gripper"
CAMERAS = ["front_cam", "side_cam"]

# API 키 로드
try:
    key_path = os.path.join(os.path.dirname(os.getcwd()), "api_key.txt")
    with open(key_path, "r") as f:
        MY_API_KEY = f.readline().strip()
except FileNotFoundError:
    print(f"❌ [Error] API 키 파일을 찾을 수 없습니다.")
    sys.exit(1)

# 초기 자세
HOME_QPOS = np.array([0, 0, 0, -1.5708, 0, 1.5708, 0.732, 0.04, 0.04])
HOME_EE_POS = np.array([0.555, 0.0, 0.524])

# 제어 상수
MAX_JOINT_VELOCITY = 0.5  # 시퀀스 제어라 속도 제한을 조금 풀음
FIXED_ROTATION = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

# ---------------------------------------------------------
# 2. Gemini 두뇌 (시퀀스 플래너)
# ---------------------------------------------------------
class GeminiBrain(threading.Thread):
    def __init__(self, client, model_name, result_queue):
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.result_queue = result_queue
        self.task_queue = queue.Queue()
        self.history = [] # 이전 행동 결과 기억
        self.daemon = True
        self.start()

    def plan(self, images, current_pos, user_task):
        self.task_queue.put((images, current_pos, user_task))

    def run(self):
        while True:
            # 요청이 올 때까지 대기
            images, current_pos, user_task = self.task_queue.get()
            self._generate_plan(images, current_pos, user_task)
            self.task_queue.task_done()

    def _generate_plan(self, images, current_pos, user_task):
        try:
            curr_str = f"[{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]"
            
            # 히스토리 요약 (최근 3개만 유지)
            history_str = "\n".join(self.history[-3:]) if self.history else "None"

            prompt = f"""
            You are a Robot Motion Planner. The user gives you a high-level task.
            You must output a SEQUENCE of actions to achieve this task.
            
            [Current State]
            - Robot End-Effector Position: {curr_str}
            - Previous Actions: {history_str}
            - Images: Provided (Front/Side View)
            
            [User Task]
            "{user_task}"

            [Instructions]
            1. Analyze the images to find the target object's 3D coordinates.
            2. Break down the task into logical steps (e.g., Approach -> Pre-grasp -> Grasp -> Lift).
            3. Assign a realistic 'action_time' (in milliseconds) for each step. 
               - Fast movement: 1000-1500ms
               - Precision movement: 2000-3000ms
               - Gripper operation: 500ms
            4. 'action' can be a coordinate list [x, y, z] OR a string "gripper_open" / "gripper_close".
            
            [Visual Scale Helper]
            - Blue cube is at [0.6, 0.4, 0.02]
            - White cube is at [0.2, 0.0, 0.02]
            - Floor is z=0.0
            
            [Output Format]
            You MUST return a pure JSON list of objects. Do not use Markdown code blocks.
            Example:
            [
                {{"action": [0.6, 0.1, 0.2], "action_time": 2000, "context": "Moving above the red cube"}},
                {{"action": [0.6, 0.1, 0.025], "action_time": 1500, "context": "Descending to grasp position"}},
                {{"action": "gripper_close", "action_time": 500, "context": "Closing gripper to pick object"}},
                {{"action": [0.6, 0.1, 0.3], "action_time": 1500, "context": "Lifting the object up"}}
            ]
            """

            content_payload = [prompt, "Front View:", images[0], "Side View:", images[1]]
            
            print(f"🧠 [Gemini] 생각하는 중... (Task: {user_task})")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_payload,
                config=types.GenerateContentConfig(response_mime_type="application/json") # JSON 모드 강제
            )
            
            if response.text:
                # JSON 파싱
                text = response.text.strip()
                # 혹시 마크다운이 섞여있을 경우 제거
                text = re.sub(r'```json|```', '', text).strip()
                
                plan = json.loads(text)
                
                # 결과 큐에 전송
                self.result_queue.put(plan)
                
                # 히스토리에 추가
                self.history.append(f"Task: {user_task} -> Executed {len(plan)} steps.")
            else:
                print("⚠️ [Gemini] 응답 없음.")

        except Exception as e:
            print(f"❌ [Error] Planning Fail: {e}")
            # 에러 시 빈 리스트 전송하여 루프 해제
            self.result_queue.put([])

# ---------------------------------------------------------
# 3. IK 및 유틸리티
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
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp[:, :7], jacr[:, :7]]) 
    diag = 0.05 * np.eye(6)
    dq = J.T @ np.linalg.solve(J @ J.T + diag, error_full)
    return dq

# ---------------------------------------------------------
# 4. 입력 처리 스레드 (Non-blocking)
# ---------------------------------------------------------
def user_input_thread(input_queue):
    print("\n💬 [System] 명령을 입력하세요 (예: 'pick up the red cube')...")
    while True:
        try:
            task = input(">>> ")
            if task.strip():
                input_queue.put(task)
        except EOFError:
            break

# ---------------------------------------------------------
# 5. 메인 루프
# ---------------------------------------------------------
def main():
    client = genai.Client(api_key=MY_API_KEY)
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    # 초기화
    data.qpos[:9] = HOME_QPOS
    data.ctrl[:] = HOME_QPOS[:8]
    data.ctrl[7] = 255 # Gripper Open
    mujoco.mj_forward(model, data)

    # 통신 채널
    brain_result_queue = queue.Queue() # Gemini -> Main
    user_input_queue = queue.Queue()   # User -> Main
    
    brain = GeminiBrain(client, MODEL_NAME, brain_result_queue)
    threading.Thread(target=user_input_thread, args=(user_input_queue,), daemon=True).start()
    
    # 상태 변수
    STATE_IDLE = 0
    STATE_THINKING = 1
    STATE_EXECUTING = 2
    
    current_state = STATE_IDLE
    action_plan = []
    current_action_idx = 0
    
    # 동작 보간(Interpolation) 변수
    action_start_time = 0
    action_duration = 0
    start_pos = np.array([0,0,0])
    target_pos = np.array([0,0,0])
    current_gripper_state = 255.0 # 255: Open, 0: Close
    
    # 현재 제어 목표 (IK용)
    ctrl_target_pos = HOME_EE_POS.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            now = time.time()

            # ---------------------------
            # [FSM] 상태 머신
            # ---------------------------
            if current_state == STATE_IDLE:
                # 사용자 입력 확인
                if not user_input_queue.empty():
                    user_task = user_input_queue.get()
                    
                    # 1. 캡처
                    captured_images = []
                    for cam_name in CAMERAS:
                        renderer.update_scene(data, camera=cam_name)
                        img = Image.fromarray(renderer.render())
                        captured_images.append(img)
                    
                    # 2. Gemini에게 요청
                    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME)
                    curr_pos = data.site_xpos[site_id].copy()
                    
                    brain.plan(captured_images, curr_pos, user_task)
                    current_state = STATE_THINKING

            elif current_state == STATE_THINKING:
                # Gemini 결과 대기
                if not brain_result_queue.empty():
                    plan = brain_result_queue.get()
                    if plan:
                        action_plan = plan
                        current_state = STATE_EXECUTING
                        current_action_idx = 0
                        action_start_time = 0 # 첫 액션 트리거용
                        
                        # 계획 출력
                        print(f"\n📋 [Plan Generated] Total {len(plan)} steps:")
                        for idx, step in enumerate(plan):
                            print(f"  {idx+1}. {step['context']} (Time: {step['action_time']}ms) -> {step['action']}")
                    else:
                        print("❌ 계획 생성 실패. 다시 시도하세요.")
                        current_state = STATE_IDLE

            elif current_state == STATE_EXECUTING:
                # 현재 액션이 끝났는지, 혹은 시작해야 하는지 확인
                elapsed = (now - action_start_time) * 1000 # ms 변환

                if action_start_time == 0 or elapsed >= action_duration:
                    # 다음 액션으로 넘어감
                    if current_action_idx < len(action_plan):
                        step = action_plan[current_action_idx]
                        
                        # 새 액션 설정
                        action_duration = step['action_time']
                        action_start_time = now
                        
                        action_val = step['action']
                        print(f"▶️ Executing: {step['context']}")

                        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME)
                        start_pos = data.site_xpos[site_id].copy() # 현재 실제 위치에서 시작

                        if isinstance(action_val, list):
                            # 이동 명령
                            target_pos = np.array(action_val)
                        elif action_val == "gripper_close":
                            current_gripper_state = 0.0
                            target_pos = start_pos # 위치 유지
                        elif action_val == "gripper_open":
                            current_gripper_state = 255.0
                            target_pos = start_pos # 위치 유지
                        
                        current_action_idx += 1
                    else:
                        # 모든 액션 완료
                        print("✅ Task Completed. Waiting for new command.\n>>> ", end="", flush=True)
                        current_state = STATE_IDLE
                
                # 실행 중 (Interpolation)
                if current_state == STATE_EXECUTING:
                    # 진행률 (0.0 ~ 1.0)
                    progress = min(1.0, ((now - action_start_time) * 1000) / action_duration) if action_duration > 0 else 1.0
                    
                    # 위치 보간 (Linear Interpolation)
                    ctrl_target_pos = start_pos + (target_pos - start_pos) * progress

            # ---------------------------
            # [Control] 물리 제어
            # ---------------------------
            # IK 계산 및 적용
            dq = solve_ik(model, data, ctrl_target_pos, FIXED_ROTATION, TARGET_SITE_NAME)
            
            # 안전장치: 너무 빠른 움직임 클램핑 (시퀀스 제어라도 물리적 한계는 필요)
            dq = np.clip(dq, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)
            
            data.ctrl[:7] = data.qpos[:7] + dq
            data.ctrl[7] = current_gripper_state

            mujoco.mj_step(model, data)
            viewer.sync()

            # 시뮬레이션 타임스텝 맞추기
            real_elapsed = time.time() - step_start
            if real_elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - real_elapsed)

if __name__ == "__main__":
    main()