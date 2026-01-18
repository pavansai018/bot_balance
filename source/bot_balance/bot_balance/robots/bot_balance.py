import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

BOT_BALANCE_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path='two_wheeled_v1.usd'
    ),
    actuators={
        'wheel_acts': ImplicitActuatorCfg(
            joint_names_expr=[
                'joint_right_wheel',
                'joint_left_wheel',
            ],
            damping=None,
            stiffness=None,
        )
    }
)