import mujoco
import mujoco.viewer
import numpy as np
import time

# ---------------------------------------------------------
# 1. 설정 및 로드
# ---------------------------------------------------------
SCENE_XML = "scene.xml"
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)

# ---------------------------------------------------------
# 2. 간단한 수치적 IK 함수 (Differential Inverse Kinematics)
# ---------------------------------------------------------
def solve_ik(model, data, target_pos, body_name="hand"):
    """
    현재의 Jacobian을 활용하여 목표 위치로 가기 위한 관절 속도를 계산합니다.
    """
    # 1. 말단 장치(hand)의 ID와 현재 위치 가져오기
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    current_pos = data.xpos[body_id]
    
    # 2. 위치 에러 계산 (target - current)
    error = target_pos - current_pos
    
    # 3. Jacobian 행렬 계산 (6 x nv)
    # nv는 시스템의 자유도수 (Panda의 경우 보통 9)
    jacp = np.zeros((3, model.nv)) # Translation Jacobian
    jacr = np.zeros((3, model.nv)) # Rotation Jacobian (여기선 위치만 사용)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    
    # 4. 팔의 관절만 해당되는 부분만 슬라이싱 (7개 관절)
    # Panda의 qpos는 [joint1~7, finger1, finger2] 구조임
    J = jacp[:, :7]
    
    # 5. Pseudo-inverse를 이용한 q_delta 계산 (Damped Least Squares)
    # dq = J^T * inv(J*J^T + lambda^2 * I) * error
    diag = 0.01 * np.eye(3)
    dq = J.T @ np.linalg.solve(J @ J.T + diag, error)
    
    return dq

# ---------------------------------------------------------
# 3. 메인 실행 루프
# ---------------------------------------------------------
def main():
    print("🦾 Panda 로봇 IK 제어 시작 (MuJoCo Native)")
    
    home_qpos = model.keyframe('home').qpos.copy()
    data.qpos[:] = home_qpos
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            current_time = time.time() - start_time

            # 🎯 목표 궤적 설정
            center = np.array([0.4, 0.0, 0.4])
            target_pos = center + np.array([
                0.1 * np.cos(current_time * 2), 
                0.1 * np.sin(current_time * 2), 
                0.0
            ])

            # 1. IK 풀기
            dq = solve_ik(model, data, target_pos)
            
            # 2. 관절 위치 업데이트 (팔의 7개 관절만)
            data.qpos[:7] += dq * 0.1
            
            # 3. 그리퍼 상태 (0.04: 열림, 0.0: 닫힘)
            gripper_val = 0.04
            data.qpos[7:9] = gripper_val

            # 4. [수정] 제어값(ctrl) 할당 
            # 팔 관절 7개 입력
            data.ctrl[:7] = data.qpos[:7]
            # 그리퍼 액추에이터(actuator8) 입력. 
            # XML 주석에 따르면 0.04m가 ctrl 255에 매핑되므로 255 입력 시 열립니다.
            data.ctrl[7] = 255.0 

            # 물리 연산 및 뷰어 업데이트
            mujoco.mj_step(model, data)
            viewer.sync()

            # 시간 동기화
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
                
if __name__ == "__main__":
    main()