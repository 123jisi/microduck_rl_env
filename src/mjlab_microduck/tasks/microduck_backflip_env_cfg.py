"""Microduck back-flip (backward roll) task — attempt 1, from the roulade recipe.

Episodic policy: robot starts standing, rolls BACKWARD over the flat top of
its head, and lands back on its feet. Deployment trigger is identical to the
roulade (policy switch = the flip starts immediately; no phase clock, no
reference motion).

DESIGN BASIS: this file is a deliberate mirror of
``microduck_roulade_env_cfg.py`` (attempt 3, run 2 — the five-run lesson
arc). The ONLY load-bearing difference is the rotation direction: by this
codebase's own convention (face-down = +90° pitch = rotation about body +y),
a BACKWARD roll integrates NEGATIVE body ω_y. All per-env state, gates, and
rewards live in mdp.py's ``backflip_*`` namespace, which is a sign-flipped
copy of the proven ``roulade_*`` machinery:

  • ONE dense progress signal — paid increments of the max-so-far cumulative
    BACKWARD rotation (potential-based: full flip pays 2π worth total,
    camping anywhere pays zero per step).
  • Landing rewards gated on FLIP COMPLETION (rotation frontier ≥ ~260°)
    plus the over-the-head contact latch — "do nothing" earns nothing, the
    standing spawn cannot farm them.
  • Reverse curriculum via mid-roll spawns pitched BACKWARD (negative
    pitch), tucked, with backward angular momentum; the accumulator is
    pre-set to the spawn |angle| so progress accounting stays consistent.
  • Support gate (accumulator only integrates while touching the ground),
    paid-rate cap (faster than the cap forfeits the excess), overspeed tax,
    sagittal flatness gate — all copied from the roulade unchanged, because
    they are direction-agnostic.

BACK-FLIP SPECIFIC NOTES:
  • The exit is over the BACK instead of the chest. The completion-gated
    landing stack (composite / upright / height / sharp / stand tax /
    rise velocity) only reads the rotation frontier, so it needs no sign
    change — it fires on the same 260°–330° window.
  • A grounded back roll must make head contact while moving backward in the
    sagittal plane.  Flat-head-top orientation is rewarded as style shaping,
    but is not a hard completion veto (that forward-roll calibration rejected
    valid backward rolls and left the whole landing stack at zero).
  • Mirror loss: the back flip is sagittal / left-right symmetric exactly
    like the roulade, so the same 61-dim mirror table applies.

Élan hook for later: reset_backflip_state.backward_vel_range gives standing
spawns an initial backward base velocity — widen it to e.g. (0.0, 0.3) to
train flips out of a backward walk. (0, 0) = standstill-only.
"""

import math
from copy import deepcopy

# Symmetry — the back flip is sagittal / left-right symmetric, same as the
# roulade; the mirror loss directly fights the sideways-collapse failure mode
# that the roulade's run 2 taught us about.
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to roulade/standup/velocity for parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a push mid-flip is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the roulade env) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: a CONTROLLED back flip takes ~2 s + rise ~1.5 s + settle (same
# timescale as the roulade — measured natural tumble 3.5–5.5 rad/s).
EPISODE_LENGTH_S = 5.0

# Empirically-measured standing trunk height (shared with roulade/standup —
# do not guess, see the standup lesson in AGENTS.md).
STAND_Z = 0.115

# ── Élan (run-up) hook ───────────────────────────────────────────────────────
# (0, 0) = flip from a standstill (same starting point as roulade run 1).
# Widen to e.g. (0.0, 0.3) to train flips entered with backward momentum.
BACKFLIP_BACKWARD_VEL_RANGE = (0.0, 0.0)

# ── Mid-roll spawn (reverse curriculum) ─────────────────────────────────────
# 90° = balanced on the head, 180° = on the back, ~340° ≈ seated leaning
# forward after a BACKWARD transit. Same window the roulade run-3 widened to
# (50°–340°): the second half of the flip (supine → seated → rise) must get
# dense on-policy data from day 0 — it IS the face-up recovery problem.
MIDROLL_PITCH_MIN   = math.radians(50.0)
MIDROLL_PITCH_MAX   = math.radians(340.0)
MIDROLL_OMEGA_RANGE = (0.0, 3.0)   # rad/s backward momentum at spawn
# Tuck anchor: legs folded (crouch-anchor values from the velstand crouch
# reset) + CHIN TUCK (it makes the head transit safer and earns the head-pivot
# style reward: neck_pitch −1 / head_pitch +1 puts the flat head top squarely
# on the floor). Identical to
# the roulade's TUCK_OVERRIDES — the tucked configuration is direction-
# agnostic; only the pitch SIGN differs, and that lives in mdp.py's reset.
TUCK_OVERRIDES = {
    2:  -1.15,  # left  hip_pitch
    3:   1.25,  # left  knee
    4:   1.05,  # left  ankle
    5:  -1.0,   # neck_pitch  (chin tuck)
    6:   1.0,   # head_pitch  (chin tuck)
    11:  1.15,  # right hip_pitch
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

# Rotation thresholds (rad) for the state-based gates — same windows the
# roulade converged on: >260° opens the landing gate, 180°–260° opens the
# rise-velocity bootstrap.
LANDING_GATE_LO = math.radians(260.0)
LANDING_GATE_HI = math.radians(330.0)
RISE_GATE_LO    = math.radians(180.0)
RISE_GATE_HI    = math.radians(260.0)

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_backflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck back-flip environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Head-ground contact — the flip's pivot signal. jaw_soft is the body that
    # carries the head collision geoms (top_head_shell = the flat top, jaw,
    # bottom_head_shell) in the standup/allcollisions model family. NAME IS
    # LOAD-BEARING: _update_backflip_accum reads it for the over-the-head
    # latch.
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Whole-robot ground contact — the SUPPORT GATE: the rotation accumulator
    # only integrates while some robot geom touches the terrain, so ballistic
    # flips ("breakdance") earn no progress and never complete.
    # NAME IS LOAD-BEARING: _update_backflip_accum reads it.
    robot_ground_cfg = ContactSensorCfg(
        name="robot_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, head_ground_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: back-flip task set ───────────────────────────────────────────
    # Progress increments — the one dense task signal during the flip. Same
    # weights the roulade run-4 converged on; the paid-rate cap matches the
    # measured natural tumble of this 10 cm robot (3.5–5.5 rad/s over the top).
    cfg.rewards["backflip_progress"] = RewardTermCfg(
        func=microduck_mdp.backflip_progress,
        weight=8.0,
        params={"target_angle": 2 * math.pi, "max_paid_rate": 5.0},
    )

    # Whip-speed tax — same 7 rad/s threshold as roulade run-4 (above the
    # measured p90 transit speed of ~5.5): taxes genuine whips, not the
    # natural tumble.
    cfg.rewards["backflip_overspeed"] = RewardTermCfg(
        func=microduck_mdp.backflip_overspeed_penalty,
        weight=-0.1,
        params={"omega_max": 7.0},
    )

    # Head-as-pivot shaping: contact × mid-flip window × backward-rate factor
    # (the rate factor kills the "rest face-down with head on floor" farm).
    cfg.rewards["backflip_head_pivot"] = RewardTermCfg(
        func=microduck_mdp.backflip_head_pivot,
        weight=0.5,
        params={
            "sensor_name": head_ground_cfg.name,
            "angle_lo": math.radians(30.0),
            "angle_hi": math.radians(240.0),
            "rate_norm": 2.0,
        },
    )

    # Clean-support standing annuity — paid only with both feet down and the
    # head currently clear. Include the neck so it cannot remain folded as a
    # third support after the roll.
    cfg.rewards["backflip_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing_composite,
        weight=4.0,
        params={
            "target_height":    STAND_Z,
            "height_std":       0.04,
            "upright_std":      0.40,
            "pose_std":         0.40,
            "joint_indices":    _LEG_JOINTS + _NECK_JOINTS,
            "gate_lo":          LANDING_GATE_LO,
            "gate_hi":          LANDING_GATE_HI,
            "target_overrides": None,
        },
    )

    # Completion-gated bootstrap layers (gradient far from the goal, where the
    # composite product is ≈0): linear upright + broad height Gaussian.
    cfg.rewards["backflip_upright_after_roll"] = RewardTermCfg(
        func=microduck_mdp.backflip_upright_after_roll,
        weight=0.75,
        params={"gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    cfg.rewards["backflip_height_after_roll"] = RewardTermCfg(
        func=microduck_mdp.backflip_height_after_roll,
        weight=0.5,
        params={
            "target_height": STAND_Z,
            "std":           0.04,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
        },
    )

    # Sharp landing layer (roulade run-4 lesson): tight-std upright × height
    # product on top of the broad composite — the ~10× differential across
    # the last mile that breaks the 27°-lean / 1-cm-crouch end basin.
    cfg.rewards["backflip_landing_sharp"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing_sharp,
        weight=2.0,
        params={
            "target_height": STAND_Z,
            "height_std":    0.010,
            "upright_std":   0.20,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
        },
    )

    # Completion-gated stand tax (THE standup/roulade lesson): once the
    # rotation is done, every step spent below STAND_Z costs — "crumple in a
    # heap after the flip" is net-negative, not merely worse-than-standing.
    # SELF-NEGATING func → POSITIVE weight (penalty sign convention).
    cfg.rewards["backflip_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.backflip_stand_tax,
        weight=2.0,
        params={
            "target_height": STAND_Z,
            "shortfall_scale": 0.02,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
        },
    )

    # The historical head latch proves a real grounded roll occurred; this
    # current-contact cost makes keeping the head planted after completion
    # strictly unattractive. The function is a non-negative cost.
    cfg.rewards["backflip_head_contact_after_roll"] = RewardTermCfg(
        func=microduck_mdp.backflip_head_contact_after_roll_penalty,
        weight=-2.0,
        params={
            "sensor_name": head_ground_cfg.name,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
        },
    )

    # Hard final-state attractor. Bootstrap terms above still provide gradient
    # while recovering; this bonus distinguishes truly standing from a lean or
    # deep crouch, and uses the same clean-support gate as the landing annuity.
    cfg.rewards["backflip_standing_success"] = RewardTermCfg(
        func=microduck_mdp.backflip_standing_success_bonus,
        weight=4.0,
        params={
            "target_height": STAND_Z,
            "height_tol": 0.0075,
            "upright_threshold": math.cos(math.radians(15.0)),
            "pose_tol": 0.25,
            "joint_indices": _LEG_JOINTS + _NECK_JOINTS,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
            "target_overrides": None,
        },
    )

    # Exit-rise bootstrap: upward CoM velocity, gated to the late-flip region
    # (supine → up is the face-up-recovery problem; end-state rewards have
    # zero gradient at zero motion there).
    cfg.rewards["backflip_rise_velocity"] = RewardTermCfg(
        func=microduck_mdp.backflip_rise_velocity,
        weight=0.75,
        params={
            "max_height": STAND_Z + 0.01,
            "gate_lo":    RISE_GATE_LO,
            "gate_hi":    RISE_GATE_HI,
        },
    )

    # Straightness — the roulade's shoulder-roll cheat has a back-flip twin
    # (rolling over a shoulder to avoid the fully-inverted configuration).
    # The accumulator's flatness gate makes side rolls unprofitable; these
    # penalties provide the dense per-step gradient back toward the plane.
    cfg.rewards["backflip_sagittal"] = RewardTermCfg(
        func=microduck_mdp.backflip_sagittal_penalty,
        weight=-0.25,
    )
    cfg.rewards["backflip_lateral_vel"] = RewardTermCfg(
        func=microduck_mdp.backflip_lateral_velocity_penalty,
        weight=-0.5,
    )
    cfg.rewards["backflip_flatness"] = RewardTermCfg(
        func=microduck_mdp.backflip_flatness_penalty,
        weight=-0.5,
    )

    # The trunk-only plane checks above cannot see an articulated cheat where
    # the head yaws/rolls and the legs sweep around the side. Keep only the
    # lateral DOFs near HOME; pitch joints remain free for the required tuck.
    cfg.rewards["backflip_lateral_joint_deviation"] = RewardTermCfg(
        func=microduck_mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    "left_hip_yaw",
                    "left_hip_roll",
                    "head_yaw",
                    "head_roll",
                    "right_hip_yaw",
                    "right_hip_roll",
                ),
            )
        },
    )

    # A clean sagittal roll carries both feet over the head together. Mirror
    # consistency prevents one leg from becoming a side-sweeping pivot while
    # leaving the symmetric pitch/knee/ankle tuck available.
    cfg.rewards["backflip_leg_symmetry"] = RewardTermCfg(
        func=microduck_mdp.leg_symmetry_reward,
        weight=0.5,
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    # Motion-blockers stay near zero during discovery (the flip IS a large
    # angular-velocity + impact event); the settle/polish pressure comes from
    # the LATE-introduced gated terms below (arrival_damping, |a_z|, torque
    # rate) — the standup timing lesson.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002   # must stay ≈0: the flip IS ω
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    # Arrival damper — trunk ω_xy² gated on standing height AND low tilt, so
    # the flip itself is never taxed; introduced at 0 and ramped by curriculum.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low":    0.09,
            "height_high":   0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # |a_z| impact shaping — active from step 0 (roulade run-1 lesson: zero
    # impact cost bred a violent whip solution; discovery is easy in this env,
    # so shaping the style from the start is the priority).
    # NOTE: trunk_vertical_accel_penalty is SELF-NEGATING (returns -|a_z|) →
    # POSITIVE weight (penalty sign convention).
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Self-collision — LIGHT: a tucked flip needs body-on-body contact
    # (knees against trunk); standup's -1.0 would fight the tuck.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Always-on upright would oppose the flip (the old attempt's core failure);
    # landing uprightness is handled by the completion-gated terms above.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / standup / roulade) ────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command obs slots: zero padding for BOTH head (4) and body (6) — the head
    # is part of the task (it's the pivot), so no head_pose command here, but
    # the 61D obs layout parity with velocity/standup/roulade is kept so the
    # runtime stack works unchanged (send zeros).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: tiny noise around zero (kept for obs-shape parity) ──────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ──────────────────────────────────────────────────────────
    # Falling over is the task — keep only the NaN guard + timeout.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Standing start + mid-backward-roll reverse-curriculum spawns; also
    # resets the rotation accumulator (must run after reset_robot_joints —
    # dict insertion order — since mid-roll tuck lerps FROM the HOME pose it
    # wrote).
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_state,
        mode="reset",
        params={
            "standing_prob":      0.5,
            "midroll_prob":       0.5,
            "standing_z_min":     0.11,
            "standing_z_max":     0.12,
            "standing_tilt_max":  math.radians(5.0),
            "backward_vel_range": BACKFLIP_BACKWARD_VEL_RANGE,
            "midroll_pitch_min":  MIDROLL_PITCH_MIN,
            "midroll_pitch_max":  MIDROLL_PITCH_MAX,
            "midroll_z_min":      0.05,
            "midroll_z_max":      0.10,
            "midroll_omega_range": MIDROLL_OMEGA_RANGE,
            "tuck_overrides":     TUCK_OVERRIDES,
            "tuck_factor_range":  (0.3, 1.0),
            "joint_noise_std":    0.08,
        },
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Reverse-curriculum mix: heavy mid-roll early (the completion sub-task is
    # learnable from day 0 — it overlaps face-up recovery), shift toward
    # standing starts as the full flip gets discovered. Mid-roll never goes to
    # zero: it keeps the second half practiced and is realistic DR anyway.
    # Same pacing as roulade run-3 (stages at 3000/6000 — run 2 shifted away
    # from mid-roll BEFORE standing-spawn flips were mastered: curriculum-
    # pacing failure).
    cfg.curriculum["backflip_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_backflip_state",
            "param_stages": [
                {"step": 0,          "params": {"standing_prob": 0.50, "midroll_prob": 0.50}},
                {"step": 3000 * 24,  "params": {"standing_prob": 0.65, "midroll_prob": 0.35}},
                {"step": 6000 * 24,  "params": {"standing_prob": 0.80, "midroll_prob": 0.20}},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # action_rate ramp — roulade run-4 values: ceiling softened -0.6 → -0.4
    # and the -0.4 stage pushed 2000 → 3000 (tightening was squeezing the
    # rise). -0.1 minimum from step 0 (run 1 bred violence under near-zero
    # smoothing).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 1500 * 24,  "weight": -0.2},
                {"step": 3000 * 24,  "weight": -0.4},
            ],
        },
    )

    # Smoothness polish — introduced only after the flip skill exists (standup
    # timing lesson: any attempt-tax active during discovery prevents the
    # maneuver from being found at all; fix is timing, not magnitude).
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "arrival_damping",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -0.025},
                {"step": 3500 * 24,  "weight": -0.05},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -5e-4},
                {"step": 3500 * 24,  "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weights: the func is self-negating (returns -|a_z|).
            "reward_name":   "gentle_landing",
            "weight_stages": [
                {"step": 0,          "weight": 0.002},
                {"step": 2500 * 24,  "weight": 0.005},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckBackFlipRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_backflip",
    run_name="microduck_backflip",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
