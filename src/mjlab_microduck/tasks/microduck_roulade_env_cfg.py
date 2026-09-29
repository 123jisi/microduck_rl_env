"""Microduck forward-roll (roulade) task — attempt 3, run 2.

Episodic policy: robot starts standing, rolls forward over the flat top of
its head, and lands back on its feet. Triggered at deployment like sit/standup
(policy switch = roll starts immediately; no phase clock, no reference motion).

RUN-2 REWORK (run 1 learned a violent ballistic "breakdance" whip — optimal
under the run-1 rewards: same 2π, sooner, no cost): rotation now only counts
while the robot touches the ground (support-gated accumulator — a roulade
never leaves the floor), the landing annuity requires an over-the-head
contact latch, paid progress rate is capped at 3 rad/s (faster forfeits the
excess), an overspeed penalty taxes |ω| > 4 rad/s, and the impact/smoothness
penalties are active from step 0 (discovery in this env is easy; style is
the scarce resource, not exploration).

Design (see the roulade section of mdp.py for the full history):
  • ONE dense progress signal — paid increments of the max-so-far cumulative
    forward rotation (potential-based: full roll pays 2π worth total, camping
    anywhere pays zero per step).
  • Landing rewards gated on ROLL COMPLETION (rotation frontier ≥ ~260°), not
    on a clock — "do nothing" earns nothing, the standing spawn cannot farm
    them, and no upright/height pressure ever opposes the flip.
  • Reverse curriculum via mid-roll spawns (the trick that fixed face-up
    recovery in standup): a slice of episodes starts 50°–185° into the roll,
    tucked, with forward angular momentum, accumulator pre-set to the spawn
    angle. The second half of a roulade IS the face-up recovery problem, which
    we know is learnable.
  • Élan hook for later: reset_roulade_state.forward_vel_range gives standing
    spawns an initial forward base velocity — set ROULADE_FORWARD_VEL_RANGE
    to e.g. (0.0, 0.3) to train rolls out of a walk. (0, 0) = standstill-only.

DR / obs / regularisers mirror the standup env (velocity sim2real parity),
with the motion-blockers (body_ang_vel, |a_z|, arrival damping) kept near zero
during discovery and introduced late by curriculum — the roll IS a large
angular-velocity, large-impact event; taxing attempts prevents discovery
(proven twice on standup).
"""

import math
from copy import deepcopy

# Symmetry — the roll is sagittal / left-right symmetric; the mirror loss
# directly fights the sideways-collapse failure seen in run 2. Enabled after
# migrating symmetry.py to the 61-dim layout (2026-08-13, includes the
# "policy" → "actor" output-key fix; roulade is the first env to use it).
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to standup/velocity for sim2real parity) ───
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # a push mid-roll is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the standup env) ───────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: a CONTROLLED roll takes ~2 s + rise ~1.5 s + settle. Run-3: 4 → 5 s
# (4 s left no room for the rise after a paced roll).
EPISODE_LENGTH_S = 5.0

# Empirically-measured standing trunk height (standup lesson: don't guess).
STAND_Z = 0.115

# ── Élan (run-up) hook ────────────────────────────────────────────────────────
# (0, 0) = roll from a standstill (run 1). Widen to e.g. (0.0, 0.3) to train
# rolls entered with forward momentum — standing spawns then get a random
# initial forward base velocity, approximating a hand-off from the walking
# policy without simulating the walk itself.
ROULADE_FORWARD_VEL_RANGE = (0.0, 0.0)

# ── Mid-roll spawn (reverse curriculum) ───────────────────────────────────────
# 90° = balanced on the head, 180° = on the back, 270° = supine, ~340° = seated
# leaning back, >260° opens the landing gate. Run-3 change: MAX widened
# 185° → 340° — run-2 wandb showed the second half of the roll (supine →
# seated → rise) was never spawned and never learned; spawns past ~300° open
# the landing gate at birth, giving dense on-policy data on the crouch→stand
# last mile (the velstand run-5 crouch-basin lesson).
MIDROLL_PITCH_MIN   = math.radians(50.0)
MIDROLL_PITCH_MAX   = math.radians(340.0)
MIDROLL_OMEGA_RANGE = (0.0, 3.0)   # rad/s forward momentum at spawn
# Tuck anchor: legs folded (crouch-anchor values from the velstand crouch
# reset) + CHIN TUCK (run-5: neck_pitch −1 / head_pitch +1 puts the flat head
# top squarely on the floor — measured axis_z −0.99 vs +0.6 for the passive
# face-plant; the head-top latch requires this, so mid-roll spawns must
# demonstrate the tucked configuration). Servo-index keyed; mid-roll spawns
# lerp HOME→tuck by a per-env factor.
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

# Rotation thresholds (rad) for the state-based gates.
LANDING_GATE_LO = math.radians(260.0)
LANDING_GATE_HI = math.radians(330.0)
RISE_GATE_LO    = math.radians(180.0)
RISE_GATE_HI    = math.radians(260.0)

# A completed roll may leave the centre of mass outside the final support
# polygon. Keep final two-foot/stillness rewards cheap for 0.35 s so the duck
# can take one or two corrective steps, then restore them over 0.65 s. The
# broad recovery stack stays active throughout; after 1.0 s the full annuity
# makes continued shuffling strictly expensive.
POSTROLL_BALANCE_GRACE_S = 0.35
POSTROLL_SETTLE_RAMP_S = 0.65
POSTROLL_INITIAL_SETTLE_SCALE = 0.10

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


def make_microduck_roulade_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck forward-roll environment configuration."""

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

    # Head-ground contact — the roll's pivot signal. jaw_soft is the body that
    # carries the head collision geoms (top_head_shell = the flat top, jaw,
    # bottom_head_shell) in robot_allcollisions.xml. NAME IS LOAD-BEARING:
    # _update_roulade_accum reads it for the over-the-head latch.
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Whole-robot ground contact — the SUPPORT GATE (run-2 fix): the rotation
    # accumulator only integrates while some robot geom touches the terrain,
    # so ballistic flips ("breakdance") earn no progress and never complete.
    # NAME IS LOAD-BEARING: _update_roulade_accum reads it.
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

    # ── Rewards: roulade task set ─────────────────────────────────────────────
    settle_gate_params = {
        "balance_grace_s": POSTROLL_BALANCE_GRACE_S,
        "settle_ramp_s": POSTROLL_SETTLE_RAMP_S,
        "initial_settle_scale": POSTROLL_INITIAL_SETTLE_SCALE,
    }

    # Progress increments — the one dense task signal during the roll. During
    # a 1.5 s roll it averages ~0.7/step; total payout per full roll from a
    # standing spawn ≈ weight × (episode steps it took) × mean ≈ weight × 50.
    cfg.rewards["roulade_progress"] = RewardTermCfg(
        func=microduck_mdp.roulade_progress,
        weight=8.0,
        # max_paid_rate: run-4 raised 3 → 5 rad/s. Measured physics (run-3
        # checkpoint eval): the over-the-top transit runs at 3.5–5.5 rad/s —
        # this robot is 10 cm tall, its natural tumble timescale is fast, and
        # the 3 rad/s cap was forfeiting most of the physically-necessary
        # rotation. Style pressure lives in |a_z| / action_rate / the support
        # gate, not in fighting gravity's clock.
        params={"target_angle": 2 * math.pi, "max_paid_rate": 5.0},
    )

    # Whip-speed tax — run-4 threshold 4 → 7 rad/s (above the measured p90
    # transit speed of ~5.5): taxes genuine whips, not the natural tumble.
    cfg.rewards["roulade_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roulade_overspeed_penalty,
        weight=-0.1,
        params={"omega_max": 7.0},
    )

    # Head-as-pivot shaping: contact × mid-roll window × forward-rate factor ×
    # continuous head-top alignment. The previous binary alignment paid a
    # face-plant 30% but gave no direction toward the hard latch pose.
    cfg.rewards["roulade_head_pivot"] = RewardTermCfg(
        func=microduck_mdp.roulade_head_pivot,
        weight=1.0,
        params={
            "sensor_name": head_ground_cfg.name,
            "angle_lo": math.radians(30.0),
            "angle_hi": math.radians(240.0),
            "rate_norm": 2.0,
        },
    )

    # Recovery bridge: completion-gated but deliberately NOT clean-gated.  A
    # low/head-supported state must see a continuous direction toward standing;
    # otherwise the post-roll taxes below make stopping before completion the
    # safest local optimum.  The clean-support annuity remains larger.
    cfg.rewards["roulade_recovery_composite"] = RewardTermCfg(
        func=microduck_mdp.roulade_recovery_composite,
        weight=2.0,
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

    # Explicit bridge across the failure point seen in the 6000-iteration
    # rollout: every standing-start episode contacted the head in the latch
    # window, but none put the top of the head down. This pays once, only when
    # the authentic latch is newly earned; late reverse-curriculum starts are
    # pre-marked paid and cannot farm it.
    cfg.rewards["roulade_head_latch"] = RewardTermCfg(
        func=microduck_mdp.roulade_head_latch_bonus,
        weight=2.0,
    )

    # Clean-support standing annuity — the dominant final-state attractor. It
    # stays closed through the 260°→330° recovery interval: the previous
    # smooth gate paid almost the whole annuity near 326° and taught the duck
    # to freeze before true completion. After 330° it starts at 10%, allowing
    # corrective steps, then reaches full strength within one second.
    cfg.rewards["roulade_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.roulade_landing_composite,
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
            **settle_gate_params,
        },
    )

    # Completion-gated bootstrap layers (gradient far from the goal, where the
    # composite product is ≈0): linear upright + broad height Gaussian.
    cfg.rewards["roulade_upright_after_roll"] = RewardTermCfg(
        func=microduck_mdp.roulade_upright_after_roll,
        weight=1.5,
        params={"gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    cfg.rewards["roulade_height_after_roll"] = RewardTermCfg(
        func=microduck_mdp.roulade_height_after_roll,
        weight=1.0,
        params={
            "target_height": STAND_Z,
            "std":           0.04,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
        },
    )

    # Sharp landing layer (run-4): tight-std upright × height product on top
    # of the broad composite. Run-3 eval showed EVERY completed episode
    # parking at the same z≈0.105 / 27°-lean pose — the broad stds score ~0.5
    # there, no gradient to finish. Sharp layer: ~0.1 at the basin, ~1.0
    # upright — 10× differential across the last mile.
    cfg.rewards["roulade_landing_sharp"] = RewardTermCfg(
        func=microduck_mdp.roulade_landing_sharp,
        weight=2.0,
        params={
            "target_height": STAND_Z,
            "height_std":    0.010,
            "upright_std":   0.20,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
            **settle_gate_params,
        },
    )

    # Completion-gated stand tax (run-3, THE standup lesson): once the
    # rotation is done, every step spent below STAND_Z costs — "crumple in a
    # heap after the roll" flips from free to net-negative, the same fix that
    # broke standup's static-sit basin (its height L1 at ÷4-scaled weight
    # 7.5). Gate closed during the roll, so the roll itself is never taxed;
    # mid/late-roll spawns are born with it active, which is the point.
    # Start near the effective magnitude of the old metre-valued tax.  The
    # normalized full-strength tax is introduced only after the roll/recovery
    # skill has had time to form (see the curriculum below).
    cfg.rewards["roulade_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.roulade_stand_tax,
        weight=0.25,
        params={
            "target_height": STAND_Z,
            "shortfall_scale": 0.02,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
        },
    )

    # The historical head latch proves that the head was used during the roll;
    # this current-contact cost prevents keeping it planted afterwards.
    cfg.rewards["roulade_head_contact_after_roll"] = RewardTermCfg(
        func=microduck_mdp.roulade_head_contact_after_roll_penalty,
        weight=-0.25,
        params={
            "sensor_name": head_ground_cfg.name,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
        },
    )

    # Smooth bridge from the current wobbling solution to a settled stand.
    # Widths are based on the model_6000 probe (vxy≈0.26 m/s, trunk ω≈0.85,
    # neck qdot≈3.1 rad/s), so the old policy scores visibly rather than seeing
    # an all-zero objective. Clean support makes this an opportunity reward:
    # balancing steps are allowed, but settling sooner earns more total annuity.
    cfg.rewards["roulade_settled_standing"] = RewardTermCfg(
        func=microduck_mdp.roulade_settled_standing_score,
        weight=2.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.02,
            "upright_std": 0.25,
            "pose_std": 0.30,
            "joint_indices": _LEG_JOINTS + _NECK_JOINTS,
            "neck_joint_indices": _NECK_JOINTS,
            "planar_speed_std": 0.30,
            "trunk_ang_vel_std": 1.0,
            "neck_speed_std": 3.0,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
            "target_overrides": None,
            **settle_gate_params,
        },
    )

    # Hard final-state attractor: correct all-joint HOME pose, clean two-foot
    # support, and low residual body/head speed. A momentary two-foot touch
    # while the head swings through the target does not count as success.
    cfg.rewards["roulade_standing_success"] = RewardTermCfg(
        func=microduck_mdp.roulade_standing_success_bonus,
        weight=4.0,
        params={
            "target_height": STAND_Z,
            "height_tol": 0.0075,
            "upright_threshold": math.cos(math.radians(15.0)),
            "pose_tol": 0.25,
            "joint_indices": _LEG_JOINTS + _NECK_JOINTS,
            "neck_joint_indices": _NECK_JOINTS,
            "max_planar_speed": 0.08,
            "max_trunk_ang_vel": 0.35,
            "max_neck_speed": 0.75,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
            "target_overrides": None,
            **settle_gate_params,
        },
    )

    # Reachable stepping stone: getting onto two clean feet in a broadly
    # correct pose is rewarded before the strict 7.5 mm / 15 deg / 0.25 rad
    # target becomes attainable.  The strict bonus above remains twice as
    # valuable and is what ultimately selects a settled HOME pose.
    cfg.rewards["roulade_intermediate_standing"] = RewardTermCfg(
        func=microduck_mdp.roulade_standing_success_bonus,
        weight=2.0,
        params={
            "target_height": STAND_Z,
            "height_tol": 0.015,
            "upright_threshold": math.cos(math.radians(25.0)),
            "pose_tol": 0.60,
            "joint_indices": _LEG_JOINTS + _NECK_JOINTS,
            "neck_joint_indices": _NECK_JOINTS,
            "max_planar_speed": 0.25,
            "max_trunk_ang_vel": 2.0,
            "max_neck_speed": 3.0,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
            "target_overrides": None,
            **settle_gate_params,
        },
    )

    # Exit-rise bootstrap: upward CoM velocity, gated to the late-roll region
    # (supine → up is the face-up-recovery problem; end-state rewards have zero
    # gradient at zero motion there — standup lesson #2).
    cfg.rewards["roulade_rise_velocity"] = RewardTermCfg(
        func=microduck_mdp.roulade_rise_velocity,
        weight=0.75,
        params={
            "max_height": STAND_Z + 0.01,
            "gate_lo":    RISE_GATE_LO,
            "gate_hi":    RISE_GATE_HI,
        },
    )

    # Straightness — run-5: the run-4 policy rolled over the SHOULDER (lower
    # energy path than straight over the head — it avoids the fully-inverted
    # configuration, same cheat human beginners default to). The structural
    # fix is the flatness gate on the accumulator + the head-top latch (side
    # rolls no longer count as rotation at all); these penalties provide the
    # dense per-step gradient back toward the plane, weights raised 5× from
    # the run-2 values that were noise against progress@8.
    cfg.rewards["roulade_sagittal"] = RewardTermCfg(
        func=microduck_mdp.roulade_sagittal_penalty,
        weight=-0.1,
    )
    cfg.rewards["roulade_lateral_vel"] = RewardTermCfg(
        func=microduck_mdp.roulade_lateral_velocity_penalty,
        weight=-0.5,
    )
    cfg.rewards["roulade_flatness"] = RewardTermCfg(
        func=microduck_mdp.roulade_flatness_penalty,
        weight=-0.5,
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    # Motion-blockers stay near zero during discovery (the roll IS a large
    # angular-velocity + impact event); the settle/polish pressure comes from
    # the LATE-introduced gated terms below (arrival_damping, |a_z|, torque
    # rate) — the standup timing lesson.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.roulade_joint_torque_rate_l2,
        weight=0.0,
        params={"gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002   # must stay ≈0: the roll is ω
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    # Arrival damper — completion gate is structural, in addition to height and
    # tilt. The old height/tilt-only gate was active at the initial upright
    # pose, so its late curriculum ramp taxed the angular velocity needed to
    # start the roll.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.roulade_arrival_damping,
        weight=0.0,
        params={
            "height_low":    0.09,
            "height_high":   0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "gate_lo":       LANDING_GATE_LO,
            "gate_hi":       LANDING_GATE_HI,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # |a_z| impact shaping on the EXIT landing only. A whole-body roll must
    # create vertical acceleration at the head pivot; charging for it before
    # completion turns a style term into an attempt tax.
    # NOTE: trunk_vertical_accel_penalty is SELF-NEGATING (returns -|a_z|) →
    # POSITIVE weight (penalty sign convention; a negative weight here would
    # reward violence — caught in the run-2 smoke test, sum was positive).
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.roulade_gentle_landing_penalty,
        weight=0.002,
        params={
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Self-collision — LIGHT: a tucked roll needs body-on-body contact
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

    # ── Observations (identical layout to walking / standup policies) ─────────
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
    # the 61D obs layout parity with velocity/standup is kept so the runtime
    # stack works unchanged (send zeros).
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

    # Standing start + mid-roll reverse-curriculum spawns; also resets the
    # rotation accumulator (must run after reset_robot_joints — dict insertion
    # order — since mid-roll tuck lerps FROM the HOME pose it wrote).
    cfg.events["set_roulade_state"] = EventTermCfg(
        func=microduck_mdp.reset_roulade_state,
        mode="reset",
        params={
            "standing_prob":      0.5,
            "midroll_prob":       0.5,
            "standing_z_min":     0.11,
            "standing_z_max":     0.12,
            "standing_tilt_max":  math.radians(5.0),
            "forward_vel_range":  ROULADE_FORWARD_VEL_RANGE,
            "midroll_pitch_min":  MIDROLL_PITCH_MIN,
            "midroll_pitch_max":  MIDROLL_PITCH_MAX,
            # Bias reverse-curriculum samples toward the head-pivot segment;
            # late recovery starts remain present and are pre-latched only
            # when they start beyond the real 170° latch opportunity.
            "midroll_pitch_power": 2.0,
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

    # Reverse-curriculum mix: keep both authentic early head-pivot starts and
    # late recovery starts, then gradually remove the early-angle sampling bias
    # as standing-start rolls take over. Mid-roll never goes to zero.
    # Run-3: stages pushed 1500/3000 → 3000/6000 — run 2 shifted away from
    # mid-roll BEFORE standing-spawn rolls were mastered (progress episode-sum
    # was ~20% of a full roll at iter 1876; curriculum-pacing failure, same
    # family as the 2026-07-28 standup regression).
    cfg.curriculum["roulade_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_roulade_state",
            "param_stages": [
                {
                    "step": 0,
                    "params": {
                        "standing_prob": 0.50,
                        "midroll_prob": 0.50,
                        "midroll_pitch_power": 2.0,
                    },
                },
                {
                    "step": 3000 * 24,
                    "params": {
                        "standing_prob": 0.60,
                        "midroll_prob": 0.40,
                        "midroll_pitch_power": 1.5,
                    },
                },
                {
                    "step": 6000 * 24,
                    "params": {
                        "standing_prob": 0.75,
                        "midroll_prob": 0.25,
                        "midroll_pitch_power": 1.0,
                    },
                },
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

    # Keep the only remaining pre-completion smoothness cost at its discovery
    # floor through the 6000-iteration horizon. Earlier runs tightened it at
    # 1500/3000 even though standing-start completion was still 0%, making the
    # already-unlearned head transition progressively less attractive.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 6500 * 24,  "weight": -0.2},
                {"step": 8500 * 24,  "weight": -0.3},
            ],
        },
    )

    # Stability costs are useful only after the dynamic skill exists.  At full
    # strength from step zero they made crossing the completion gate worth
    # roughly -5 episode reward in the failed run, while all clean-standing
    # rewards stayed near zero.  This staged schedule preserves discovery,
    # then progressively makes fast clean recovery preferable.
    cfg.curriculum["roulade_stand_tax_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "roulade_stand_tax",
            "weight_stages": [
                {"step": 0,          "weight": 0.25},
                {"step": 4000 * 24,  "weight": 0.5},
                {"step": 6500 * 24,  "weight": 1.0},
                {"step": 8500 * 24,  "weight": 2.0},
            ],
        },
    )
    cfg.curriculum["roulade_head_contact_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "roulade_head_contact_after_roll",
            "weight_stages": [
                {"step": 0,          "weight": -0.25},
                {"step": 4000 * 24,  "weight": -0.5},
                {"step": 6500 * 24,  "weight": -1.0},
                {"step": 8500 * 24,  "weight": -2.0},
            ],
        },
    )

    # Smoothness polish — introduced only after the roll skill exists (standup
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

MicroduckRouladeRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_roulade",
    run_name="microduck_roulade",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
