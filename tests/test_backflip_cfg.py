import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
    make_microduck_backflip_env_cfg,
    MicroduckBackFlipRlCfg,
)


def test_cfg_has_the_backflip_rewards():
    cfg = make_microduck_backflip_env_cfg()
    for name in (
        "backflip_progress",
        "backflip_overspeed",
        "backflip_head_pivot",
        "backflip_landing_composite",
        "backflip_upright_after_roll",
        "backflip_height_after_roll",
        "backflip_landing_sharp",
        "backflip_stand_tax",
        "backflip_rise_velocity",
        "backflip_sagittal",
        "backflip_lateral_vel",
        "backflip_flatness",
    ):
        assert name in cfg.rewards, name


def test_progress_is_the_dominant_task_reward():
    cfg = make_microduck_backflip_env_cfg()
    assert cfg.rewards["backflip_progress"].weight == 8.0
    # roulade run-4 values: paid-rate cap matches the measured natural tumble
    assert cfg.rewards["backflip_progress"].params["max_paid_rate"] == 5.0


def test_penalty_sign_convention():
    # Infallible check (AGENTS.md): every penalty must be ≤ 0 under its weight.
    cfg = make_microduck_backflip_env_cfg()
    # overspeed / sagittal / lateral_vel / flatness return positive quantities
    for name in ("backflip_overspeed", "backflip_sagittal", "backflip_lateral_vel", "backflip_flatness"):
        assert cfg.rewards[name].weight < 0.0, name
    # stand_tax is SELF-NEGATING → POSITIVE weight (double-negation trap)
    assert cfg.rewards["backflip_stand_tax"].weight > 0.0
    # gentle_landing is SELF-NEGATING → POSITIVE weight
    assert cfg.rewards["gentle_landing"].weight > 0.0


def test_always_on_upright_is_removed():
    # An always-on upright term would oppose the flip (old attempt's core
    # failure). Upright pressure must only exist as completion-gated terms.
    cfg = make_microduck_backflip_env_cfg()
    assert "upright" not in cfg.rewards
    assert "fell_over" not in cfg.terminations


def test_motion_blockers_stay_near_zero():
    cfg = make_microduck_backflip_env_cfg()
    # the flip IS a large angular-velocity + impact event — these must not
    # fight discovery
    assert abs(cfg.rewards["body_ang_vel"].weight) <= 0.002
    assert abs(cfg.rewards["angular_momentum"].weight) <= 0.001
    assert cfg.rewards["arrival_damping"].weight == 0.0
    assert cfg.rewards["joint_torque_rate_l2"].weight == 0.0


def test_reverse_curriculum_spawn_mix_paces_toward_standing():
    cfg = make_microduck_backflip_env_cfg()
    stages = cfg.curriculum["backflip_spawn_mix"].params["param_stages"]
    assert stages[0]["step"] == 0
    standing = [s["params"]["standing_prob"] for s in stages]
    midroll = [s["params"]["midroll_prob"] for s in stages]
    assert standing == sorted(standing)              # never decreases
    assert midroll == sorted(midroll, reverse=True)  # never increases
    assert standing[-1] == 0.80 and midroll[-1] == 0.20
    # same pacing as roulade run-3 (3000/6000 — run 2 shifted too early)
    assert [s["step"] for s in stages[1:]] == [3000 * 24, 6000 * 24]


def test_spawn_event_is_backward_roll():
    cfg = make_microduck_backflip_env_cfg()
    ev = cfg.events["set_backflip_state"]
    assert ev.func is microduck_mdp.reset_backflip_state
    params = ev.params
    # spawn window covers the second half of the flip (the roulade run-3 fix)
    assert params["midroll_pitch_min"] == math.radians(50.0)
    assert params["midroll_pitch_max"] == math.radians(340.0)
    # chin-tuck overrides present (the head-top latch requires the tuck)
    assert params["tuck_overrides"][5] == -1.0   # neck_pitch
    assert params["tuck_overrides"][6] == 1.0    # head_pitch


def test_symmetry_is_enabled():
    # The back flip is sagittal / left-right symmetric, same as the roulade —
    # the mirror loss fights the sideways-collapse failure mode.
    assert MicroduckBackFlipRlCfg.algorithm.symmetry_cfg is not None


def test_required_sensors_registered():
    cfg = make_microduck_backflip_env_cfg()
    names = {s.name for s in cfg.scene.sensors}
    # accumulator support gate + head latch read these by name (load-bearing)
    assert "robot_ground_contact" in names
    assert "head_ground_contact" in names
