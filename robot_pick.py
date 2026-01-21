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
# [New] 회전 오차 계산 헬퍼 함수
# ---------------------------------------------------------
def get_orientation_error(current_mat, target_mat):
    """
    현재 회전 행렬(3x3)과 목표 회전 행렬(3x3)의 차이를 
    회전 벡터(Rotation Vector)로 변환하여 반환합니다.
    """
    # 1. 회전 오차 행렬 R_err = R_target * R_current^T
    # (현재 자세에서 목표 자세로 가기 위한 회전 변환)
    r_err_mat = target_mat @ current_mat.T
    
    # 2. MuJoCo 함수를 이용해 행렬을 쿼터니언으로 변환
    quat_err = np.zeros(4)
    mujoco.mju_mat2Quat(quat_err, r_err_mat.flatten())
    
    # 3. 쿼터니언을 3D 회전 벡터(축 * 각도)로 변환 (속도 제어용)
    # 쿼터니언 q = [w, x, y, z] 일 때, 
    # 회전 벡터 v는 2 * [x, y, z] (w가 1에 가까울 때의 근사치)
    # 정확한 계산: 2 * arccos(w) * (v / sin(theta/2))
    # 여기서는 MuJoCo의 mju_quat2Vel 함수를 활용할 수도 있으나, 
    # 간단히 구현하면 다음과 같습니다.
    
    ref_quat = np.array([1.0, 0.0, 0.0, 0.0]) # 단위 쿼터니언
    rot_err = np.zeros(3)
    
    # 쿼터니언 차이를 각속도 벡터로 변환 (mju_quat2Vel 유사 기능)
    # w(스칼라)가 음수면 반대 방향으로 도는 게 빠르므로 부호 반전
    if quat_err[0] < 0:
        quat_err = -quat_err
        
    # 간단한 비례 제어용 회전 벡터 (sin(theta/2) * axis)
    rot_err = quat_err[1:] * 2.0
    
    return rot_err

# ---------------------------------------------------------
# [Updated] 6-DoF IK 함수
# ---------------------------------------------------------
def solve_ik(model, data, target_pos, target_rot, site_name="gripper"):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    
    # 1. 위치 오차 (Position Error)
    current_pos = data.site_xpos[site_id]
    error_pos = target_pos - current_pos
    
    # 2. 방향 오차 (Orientation Error)
    # site_xmat은 9개짜리 1차원 배열이므로 3x3으로 reshape 필요
    current_mat = data.site_xmat[site_id].reshape(3, 3)
    error_rot = get_orientation_error(current_mat, target_rot)
    
    # 3. 전체 에러 벡터 (6차원: 위치 3 + 회전 3)
    error_full = np.hstack([error_pos, error_rot])
    
    # 4. Jacobian 계산 (Position + Rotation)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    
    # 팔 관절(7개)만 사용
    J_pos = jacp[:, :7]
    J_rot = jacr[:, :7]
    
    # 5. 전체 Jacobian 스택 (6 x 7 행렬)
    J_full = np.vstack([J_pos, J_rot])
    
    # 6. Damped Least Squares 풀이
    # 6x6 단위 행렬에 댐핑 계수 적용
    diag = 0.05 * np.eye(6) 
    dq = J_full.T @ np.linalg.solve(J_full @ J_full.T + diag, error_full)
    
    return dq

# ---------------------------------------------------------
# 3. 메인 실행 루프
# ---------------------------------------------------------
def main():
    print("🦾 Panda 로봇 IK 제어 시작 (MuJoCo Native)")
    
    HOME_QPOS = np.array([0, 0, 0, -1.5708, 0, 1.5708, -0.7853, 0.732, 0.04])
    data.qpos[:9] = HOME_QPOS
    mujoco.mj_forward(model, data)

    target_rot = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            time.sleep(0.3)
            step_start = time.time()
            
            # 목표 위치
            target_pos = np.array([0.5, -0.2, 0.025])
            
            # 1. 6자유도 IK 풀기 (target_rot 추가)
            dq = solve_ik(model, data, target_pos, target_rot)
            
            # 2. 관절 업데이트 (속도 조절을 위해 0.5 정도 곱해줌)
            data.qpos[:7] += dq * 0.5
            
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