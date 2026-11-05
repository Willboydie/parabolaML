import numpy as np
import time
import os

# Headless mode and step cap for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "180000"))  #NEW

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing, InterceptionExpert, make_catchable_pair,  #NEW
)


# ==============================================================================
# TARGET SPECIFICATION
# Random moving target in a 5 x 5 m cuboid centred on the initial robot XY position.  #NEW
# Height limits: 0.07 to 1.30 m; high parabolic paths leave the roof and return to opposite side faces.  #NEW
from moving_target import MovingTarget  #NEW
TARGET_SEED = int(os.environ["ROBOENV_TARGET_SEED"]) if "ROBOENV_TARGET_SEED" in os.environ else None  #NEW
TARGET_SPEED = float(os.environ["ROBOENV_TARGET_SPEED"]) if "ROBOENV_TARGET_SPEED" in os.environ else None  #NEW

# Fixed gripper finger configuration (angles in radians, held fixed during simulation):
# [finger_1, finger_tip_1, finger_2, finger_tip_2, finger_3, finger_tip_3]
# Set to None to use initial configuration from config/URDF, or an array of 6 angles.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Max navigation time (s); enlarged for targets around five metres away.  #NEW
# Targets behind the robot need a large turn, so this must be generous.
BASE_NAV_TIMEOUT = 240.0  #NEW
# Max EE distance from the shoulder the arm is asked to reach (m). Targets further away are projected
# onto this sphere, so the arm settles at the closest pose instead of fully stretching (singularity).
ARM_REACH_LIMIT = 0.85
# Base-to-target horizontal standoff (m) as a function of target height: larger for low targets,
# default for mid heights, smaller for high targets (far above the shoulder, which sits ~0.49 m high).
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
# Stop once the EE is this close to the target (m); max iterations set via ROBOENV_MAX_STEPS
REACH_TOLERANCE = 0.01  #NEW
# ==============================================================================


def main(options=None):  #NEW
    options = {} if options is None else options  #NEW
    recorder = options.get("recorder")  #NEW
    episode_seed = options.get("seed", TARGET_SEED)  #NEW
    episode_speed = options.get("target_speed", TARGET_SPEED)  #NEW
    target_count = options.get("balls", 2)  #NEW
    episode_max_steps = options.get("max_steps", max_steps)  #NEW
    # --------------------------------------------------------------------------
    # Configuration and Simulation Setup
    # --------------------------------------------------------------------------
    conf_file_name = "robotnik_arm.json"
    cur_dir = options.get("assets_dir") or os.path.dirname(os.path.abspath(__file__))  #NEW
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    # Configure realistic ground contact and skid-steer friction dynamics
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    client.changeVisualShape(sim.ground_body, -1, textureUniqueId=-1, rgbaColor=[0.55, 0.58, 0.62, 1.0])  #NEW
    rid = sim.bot[0].bot_pybullet
    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0]
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    # --------------------------------------------------------------------------
    # Gripper Fingers: Fix gripper finger configuration
    # --------------------------------------------------------------------------
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    # Active joints from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]

    # Pinocchio dynamic & kinematic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    controlled_frame_name = "robot_j2s7s300_end_effector"
    ee_frame_id = dyn_model.pin_model.getFrameId(controlled_frame_name)
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())

    # Forward kinematics for initial end-effector position
    init_q_all = np.concatenate((init_base_pos, init_base_ori, init_joint_angles))
    init_q_pin = dyn_model.ReoderJoints2PinVec(init_q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, init_q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    # Joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()

    time_step = sim.GetTimeStep()
    current_time = 0

    # --------------------------------------------------------------------------
    # Coordinated Motion Planning: Base Navigation & Arm Reach
    # --------------------------------------------------------------------------
    # Standoff distance between robot base center and target, depending on target height.
    base_goal_pos = init_base_pos[:2].copy()  #NEW
    base_goal_bearing = 0.0  #NEW

    # Base navigation controller (align -> drive -> orient -> done), see BaseControllerParams
    base_state = BaseNavState()
    base_navigation_done = base_state.done
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)  #NEW

    # Arm controller (differential IK + joint PD), see ArmControllerParams
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)  #NEW
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])
    # Visualize both validated targets and their parabolic paths.  #NEW
    target_center = np.array(sim.GetBasePosition(), dtype=float)  #NEW
    target_lower = np.array([target_center[0] - 2.5, target_center[1] - 2.5, 0.07])  #NEW
    target_upper = np.array([target_center[0] + 2.5, target_center[1] + 2.5, 1.30])  #NEW
    initial_bearing = quaternion2bearing(init_base_ori[3], *init_base_ori[:3])  #NEW
    targets, selected_target, validation_plans = make_catchable_pair(episode_seed, episode_speed, target_lower, target_upper, init_base_pos, initial_bearing, init_ee_pos, time_step, base_params, arm_params, target_count=target_count)  #NEW
    target = targets[selected_target]  #NEW
    if not headless:  #NEW
        peak_times = [-candidate.velocity[2] / candidate.acceleration[2] for candidate in targets]  #NEW
        highest_arc = max(candidate.position(t)[2] for candidate, t in zip(targets, peak_times))  #NEW
        client.resetDebugVisualizerCamera(cameraDistance=max(10.0, 2.0 * highest_arc + 3.0), cameraYaw=45, cameraPitch=-40, cameraTargetPosition=[target_center[0], target_center[1], highest_arc / 2])  #NEW
    target_bodies = []  #NEW
    for index, candidate in enumerate(targets):  #NEW
        candidate.draw(client, draw_box=index == 0, path_color=[1.0, 0.65, 0.0] if index == selected_target else [0.65, 0.4, 1.0])  #NEW
        color = [1.0, 0.15, 0.0, 1.0] if index == selected_target else [0.45, 0.2, 1.0, 1.0]  #NEW
        body = candidate.spawn(client, color=color)  #NEW
        target_bodies.append(body)  #NEW
    target_pos = target.position(0.0)  #NEW
    distances = [np.linalg.norm(candidate.position(0.0) - target_center) for candidate in targets]  #NEW
    print(f"Randomly selected catchable ball {selected_target + 1}; initial 3D distances: {np.round(distances, 3)} m")  #NEW
#NEW
    expert = InterceptionExpert(target.lower, target.upper)  #NEW
    for index, plan in enumerate(validation_plans):  #NEW
        print(f"Ball {index + 1}: estimated moving catch at {plan.intercept_time:.2f}s, endpoint at {targets[index].duration:.2f}s")  #NEW
    print("==================================================================")  #NEW
    print(f"Target Cartesian EE Position : {target_pos}")  #NEW
    print(f"Target speed: {target.speed:.3f} m/s | path duration: {target.duration:.1f}s | end: {target.end}")  #NEW
    print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")  #NEW
    print(f"Initial Cartesian Error (m)  : {np.linalg.norm(target_pos - init_ee_pos):.4f}")  #NEW
    print("==================================================================")  #NEW

    # Stop on actual target contact or step cap; waiting for interception is not a stall.  #NEW
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=episode_max_steps, stall_window=np.inf, min_time=0.0)  #NEW

    if recorder is not None:  #NEW
        recorder.start(targets, selected_target, time_step, ext_names, target_lower, target_upper)  #NEW
    stop_message = "User quit"  #NEW
    # Command interface
    cmd = MotorCommands()

    # Data logging for plotting
    time_all = []
    ee_pos_all = []
    ee_error_all = []  #NEW
    target_pos_all = []  #NEW
    base_pos_all = []
    q_mes_all = []
    qd_mes_all = []
    q_d_all = []
    qd_d_all = []

    # --------------------------------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------------------------------
    planning_attempted = False  #NEW
    while True:
        for candidate, body in zip(targets, target_bodies):  #NEW
            client.resetBasePositionAndOrientation(body, candidate.position(current_time).tolist(), [0, 0, 0, 1])  #NEW
        target_pos = target.position(current_time)  #NEW
        # Measure current states  #NEW
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        # Full kinematic state vector in Pinocchio format
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")

        # Compute end-effector forward kinematics
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = target_pos - ee_pos  #NEW
        ee_dist_error = np.linalg.norm(ee_error)

        if recorder is not None and recorder.step >= episode_max_steps:  #NEW
            stop_message = f"Reached step cap {episode_max_steps} at t={current_time:.2f}s"  #NEW
            break  #NEW
        # ----------------------------------------------------------------------
        # 1. Mobile Base Control (Wheel velocities)
        # ----------------------------------------------------------------------
        ready = expert.observe(current_time, target_pos)  #NEW
        missed_intercept = expert.intercept_time is not None and current_time > expert.intercept_time + 1.0 and np.linalg.norm(target_pos - expert.intercept) > REACH_TOLERANCE  #NEW
        if ready and (not planning_attempted or missed_intercept):  #NEW
            planning_attempted = True  #NEW
            base_state = BaseNavState()  #NEW
            planned = expert.plan(current_time, target_pos, base_pos, base_bearing, ee_pos, base_params, arm_params)  #NEW
            if planned:  #NEW
                base_goal_pos, base_goal_bearing = expert.goal_pos, expert.goal_bearing  #NEW
                print(f"Parabolic interception: {expert.intercept} at t={expert.intercept_time:.2f}s")  #NEW
                client.addUserDebugPoints([expert.intercept.tolist()], [[0, 1, 0]], pointSize=12)  #NEW
            else:  #NEW
                print("No feasible interception within the planning horizon.")  #NEW
        ready = ready and expert.intercept is not None  #NEW
        wheel_cmds = base_controller(  #NEW
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, base_params  #NEW
        ) if ready else np.zeros(4)  #NEW
        base_navigation_done = base_state.done

        # ----------------------------------------------------------------------
        # 2. Arm Control (reaching via differential IK + gravity-compensated joint PD)
        # ----------------------------------------------------------------------
        # While the base is travelling the arm holds its nominal configuration.
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, expert.intercept if ready else target_pos, ee_pos,  #NEW
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
            reach_active=ready and base_navigation_done,  #NEW
        )

        # ----------------------------------------------------------------------
        # 3. Gripper / Finger Control (Fingers locked at fixed configuration)
        # ----------------------------------------------------------------------
        torque_fingers = np.zeros(6)

        # ----------------------------------------------------------------------
        # Send Multi-Interface Control Commands
        # ----------------------------------------------------------------------
        # 4 wheels (velocity) + 7 arm joints (torque) + 6 fingers (torque)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))  #NEW
        if recorder is not None:  #NEW
            state = dict(time=current_time, base_pos=base_pos.copy(), base_ori=base_ori.copy(),  #NEW
                         base_bearing=base_bearing, base_velocity=np.r_[sim.GetBaseLinVelocity(), sim.GetBaseAngVelocity()],  #NEW
                         joint_pos=q_mes.copy(), joint_vel=qd_mes.copy(), ee_pos=ee_pos.copy(),  #NEW
                         targets=np.array([candidate.position(current_time) for candidate in targets]),  #NEW
                         selected=selected_target, phase=base_state.phase, nav_done=base_navigation_done,  #NEW
                         q_des=q_des_arm.copy(), qd_des=qd_des_arm.copy(), ready=ready,  #NEW
                         intercept=expert.intercept, intercept_time=expert.intercept_time,  #NEW
                         base_goal=base_goal_pos.copy(), base_goal_bearing=base_goal_bearing,  #NEW
                         ee_error=float(ee_dist_error))  #NEW
            cmd_all = recorder.command(state, cmd_all, base_params, arm_params)  #NEW
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")
        if recorder is not None:  #NEW
            recorder.after_step(np.array(sim.GetMotorAngles(0)), np.array(sim.GetBasePosition()), np.array([candidate.position(current_time + time_step) for candidate in targets]))  #NEW

        # Visualization in Pinocchio GUI if active
        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        # Exit logic with 'q' key in PyBullet GUI window
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Check distance to the moving target rather than the predicted interception point.  #NEW
        capture_allowed = base_navigation_done and expert.has_left_volume and np.all(target_pos >= target.lower - 1e-6) and np.all(target_pos <= target.upper + 1e-6)  #NEW
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, capture_allowed)  #NEW
        if stop_message:
            print(stop_message)
            break

        # Data logging for plotting
        if headless:  #NEW
            current_time += time_step  #NEW
            continue  #NEW
        time_all.append(current_time)
        ee_pos_all.append(ee_pos)  #NEW
        target_pos_all.append(target_pos.copy())  #NEW
        ee_error_all.append(ee_error)
        base_pos_all.append(base_pos)
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        current_desired_q = np.concatenate((q_mes[:4], q_des_arm, target_finger_angles))
        current_desired_qd = np.concatenate((wheel_cmds, qd_des_arm, np.zeros(6)))
        q_d_all.append(current_desired_q)
        qd_d_all.append(current_desired_qd)

        if not headless:
            time.sleep(0.001)

        current_time += time_step
        if int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | EE pos: {np.round(ee_pos, 3)} | EE error: {ee_dist_error:.4f}m | target {np.round(target_pos, 3)} | yaw {base_bearing:.2f} phase {base_state.phase} nav_done {base_navigation_done}")  #NEW

    # Final summary report
    print("==================================================================")  #NEW
    print(f"Final Base Position          : {np.round(base_pos[:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(ee_pos, 4)}")
    print(f"Target End-Effector Position : {target_pos}")  #NEW
    print(f"Final Cartesian Error (m)    : {np.linalg.norm(target_pos - ee_pos):.4f}")  #NEW
    print(f"Fixed Finger Angles          : {np.round(target_finger_angles, 4)}")
    print(f"Final Finger Angles          : {np.round(q_mes[11:17], 4)}")
    print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
    print("==================================================================")  #NEW

    if headless:
        print(f"Headless run finished at t={current_time:.2f}s")
        if recorder is not None:  #NEW
            recorder.finish(stop_message, current_time, float(ee_dist_error))  #NEW
        client.disconnect()  #NEW
        return  #NEW

    # --------------------------------------------------------------------------
    # Plotting Performance
    # --------------------------------------------------------------------------
    ee_pos_all = np.array(ee_pos_all)
    ee_error_all = np.array(ee_error_all)  #NEW
    target_pos_all = np.array(target_pos_all)  #NEW
    time_all = np.array(time_all)

    # 1. Cartesian end-effector coordinates tracking
    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.plot(time_all, target_pos_all[:, idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')  #NEW
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching')
    plt.tight_layout()
    plt.show()

    # 2. Cartesian distance error over time
    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Arm joint position tracking
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in q_mes_all], label=f'Measured J{i+1}')
        plt.plot(time_all, [q[4 + i] for q in q_d_all], label=f'Desired J{i+1}', linestyle='--')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
