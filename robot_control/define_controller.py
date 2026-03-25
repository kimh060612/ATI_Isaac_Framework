from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from ati_config import ATIBaseRobotConfig

def define_agent_controller(
    agent_name: str,
    agent_prim_path: str,
    robot_config: ATIBaseRobotConfig = None
):
    if agent_name == "kaya":
        kaya_setup = HolonomicRobotUsdSetup(
            robot_prim_path=agent_prim_path, 
            com_prim_path=f"{agent_prim_path}/base_link/control_offset"
        )
        (
            wheel_radius,
            wheel_positions,
            wheel_orientations,
            mecanum_angles,
            wheel_axis,
            up_axis,
        ) = kaya_setup.get_holonomic_controller_params()
        
        agent_controller = HolonomicController(
            name="holonomic_controller",
            wheel_radius=wheel_radius,
            wheel_positions=wheel_positions,
            wheel_orientations=wheel_orientations,
            mecanum_angles=mecanum_angles,
            wheel_axis=wheel_axis,
            up_axis=up_axis,
        )
    elif agent_name == "limo":
        agent_controller = DifferentialController(
            name="limo_diff_controller",
            wheel_radius=robot_config.wheelRadius,
            wheel_base=robot_config.wheelDistance, 
            max_angular_speed=robot_config.maxLinearSpeed / robot_config.wheelRadius,
            max_linear_speed=robot_config.maxLinearSpeed,
            max_wheel_speed=robot_config.maxLinearSpeed / robot_config.wheelRadius
        )
    else:
        raise ValueError(f"Unsupported agent name: {agent_name}. Supported agents are 'kaya' and 'limo'.")

    return agent_controller