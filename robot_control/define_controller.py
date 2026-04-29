from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from robot_control import CircularController
from ati_config import ATIBaseRobotConfig

def define_agent_controller(
    agent_name: str,
    agent_prim_path: str,
    robot_config: ATIBaseRobotConfig = None
):
    if agent_name == "kaya":
        wheel_radius = [0.04, 0.04, 0.04]
        wheel_orientations = [[0, 0, 0, 1], [0.866, 0, 0, -0.5], [0.866, 0, 0, 0.5]]
        wheel_positions = [
            [-0.0980432, 0.000636773, -0.050501],
            [0.0493475, -0.084525, -0.050501],
            [0.0495291, 0.0856937, -0.050501],
        ]
        mecanum_angles = [90, 90, 90]
        
        agent_controller = HolonomicController(
            name="holonomic_controller",
            wheel_radius=wheel_radius,
            wheel_positions=wheel_positions,
            wheel_orientations=wheel_orientations,
            mecanum_angles=mecanum_angles,
        )
    elif agent_name == "limo":
        agent_controller = CircularController(
            name="limo_circular_controller",
            wheel_radius=robot_config.wheelRadius,
            wheel_base=robot_config.wheelDistance,
        )
        
        # DifferentialController(
        #     name="limo_diff_controller",
        #     wheel_radius=robot_config.wheelRadius,
        #     wheel_base=robot_config.wheelDistance, 
        #     max_angular_speed=robot_config.maxLinearSpeed / robot_config.wheelRadius,
        #     max_linear_speed=robot_config.maxLinearSpeed,
        #     max_wheel_speed=robot_config.maxLinearSpeed / robot_config.wheelRadius
        # )
    else:
        raise ValueError(f"Unsupported agent name: {agent_name}. Supported agents are 'kaya' and 'limo'.")

    return agent_controller