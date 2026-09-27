import math
from types import SimpleNamespace

import torch

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
        "backflip_head_contact_after_roll",
        "backflip_standing_success",
        "backflip_rise_velocity",
        "backflip_sagittal",
        "backflip_lateral_vel",
        "backflip_flatness",
        "backflip_lateral_joint_deviation",
        "backflip_leg_symmetry",
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
    for name in (
        "backflip_overspeed",
        "backflip_sagittal",
        "backflip_lateral_vel",
        "backflip_flatness",
        "backflip_lateral_joint_deviation",
    ):
        assert cfg.rewards[name].weight < 0.0, name
    # stand_tax is SELF-NEGATING → POSITIVE weight (double-negation trap)
    assert cfg.rewards["backflip_stand_tax"].weight > 0.0
    # current head-contact term returns a non-negative cost
    assert cfg.rewards["backflip_head_contact_after_roll"].weight < 0.0
    # leg_symmetry returns a self-negating penalty
    assert cfg.rewards["backflip_leg_symmetry"].weight > 0.0
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


def test_side_twist_constraints_leave_pitch_tuck_free():
    cfg = make_microduck_backflip_env_cfg()
    lateral = cfg.rewards["backflip_lateral_joint_deviation"]
    assert lateral.func is microduck_mdp.joint_deviation_l1
    names = set(lateral.params["asset_cfg"].joint_names)
    assert names == {
        "left_hip_yaw",
        "left_hip_roll",
        "head_yaw",
        "head_roll",
        "right_hip_yaw",
        "right_hip_roll",
    }
    assert not any("pitch" in name or "knee" in name or "ankle" in name for name in names)
    assert cfg.rewards["backflip_leg_symmetry"].func is microduck_mdp.leg_symmetry_reward
    assert cfg.rewards["backflip_sagittal"].weight == -0.25


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


def test_clean_stand_gate_requires_both_feet_and_current_head_clear():
    gate = microduck_mdp._backflip_clean_stand_gate_from_contacts(
        completion_gate=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        feet_contact=torch.tensor([
            [True, True],
            [True, False],
            [True, True],
            [True, True],
        ]),
        head_contact=torch.tensor([[False], [False], [True], [False]]),
    )
    assert gate.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_normalized_stand_tax_scale_and_cap():
    shortfall = microduck_mdp._normalized_height_shortfall(
        z=torch.tensor([0.115, 0.105, 0.095, 0.075]),
        target_height=0.115,
        shortfall_scale=0.02,
    )
    assert torch.allclose(shortfall, torch.tensor([0.0, 0.5, 1.0, 1.0]))


def test_landing_terms_use_clean_support_but_bootstrap_terms_do_not():
    cfg = make_microduck_backflip_env_cfg()
    assert (
        cfg.rewards["backflip_landing_composite"].func
        is microduck_mdp.backflip_landing_composite
    )
    assert (
        cfg.rewards["backflip_landing_sharp"].func
        is microduck_mdp.backflip_landing_sharp
    )
    assert (
        cfg.rewards["backflip_upright_after_roll"].func
        is microduck_mdp.backflip_upright_after_roll
    )
    assert (
        cfg.rewards["backflip_height_after_roll"].func
        is microduck_mdp.backflip_height_after_roll
    )
    assert cfg.rewards["backflip_upright_after_roll"].weight == 0.75
    assert cfg.rewards["backflip_height_after_roll"].weight == 0.5


def test_final_pose_terms_include_neck_and_use_strict_thresholds():
    cfg = make_microduck_backflip_env_cfg()
    expected_joints = list(range(14))
    assert (
        sorted(cfg.rewards["backflip_landing_composite"].params["joint_indices"])
        == expected_joints
    )
    success = cfg.rewards["backflip_standing_success"]
    assert sorted(success.params["joint_indices"]) == expected_joints
    assert success.params["height_tol"] == 0.0075
    assert success.params["upright_threshold"] == math.cos(math.radians(15.0))
    assert success.params["pose_tol"] == 0.25
    assert cfg.rewards["backflip_landing_sharp"].params["height_std"] == 0.010
    assert cfg.rewards["backflip_landing_sharp"].params["upright_std"] == 0.20
    assert cfg.rewards["backflip_stand_tax"].params["shortfall_scale"] == 0.02
    assert cfg.rewards["backflip_stand_tax"].weight == 2.0


def test_head_contact_cost_is_gated_by_completion(monkeypatch):
    class Scene:
        sensors = {
            "head_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.tensor([[1], [1], [0]]))
            )
        }

        def __getitem__(self, _name):
            return SimpleNamespace()

    env = SimpleNamespace(num_envs=3, device=torch.device("cpu"), scene=Scene())

    monkeypatch.setattr(microduck_mdp, "_update_backflip_accum", lambda _env, _asset: None)
    monkeypatch.setattr(
        microduck_mdp,
        "_backflip_completion_gate",
        lambda _env, _lo, _hi, require_head: torch.tensor([1.0, 0.0, 1.0]),
    )
    result = microduck_mdp.backflip_head_contact_after_roll_penalty(env)
    assert result.tolist() == [1.0, 0.0, 0.0]


def test_final_annuity_reads_clean_gate_while_bootstraps_read_completion(monkeypatch):
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.115]] * 2),
        )
    )

    class Scene:
        terrain = SimpleNamespace(env_origins=torch.zeros(2, 3))

        def __getitem__(self, _name):
            return asset

    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), scene=Scene())
    monkeypatch.setattr(microduck_mdp, "_update_backflip_accum", lambda _env, _asset: None)
    monkeypatch.setattr(
        microduck_mdp,
        "_backflip_clean_stand_gate",
        lambda _env, _lo, _hi: torch.tensor([1.0, 0.0]),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_backflip_completion_gate",
        lambda _env, _lo, _hi, require_head: torch.ones(2),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "standing_composite_score",
        lambda *args, **kwargs: torch.ones(2),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "standing_success_bonus",
        lambda *args, **kwargs: torch.ones(2),
    )

    composite = microduck_mdp.backflip_landing_composite(
        env, 0.115, 0.04, 0.4, 0.4, list(range(14))
    )
    sharp = microduck_mdp.backflip_landing_sharp(env, 0.115)
    success = microduck_mdp.backflip_standing_success_bonus(
        env,
        target_height=0.115,
        height_tol=0.0075,
        upright_threshold=math.cos(math.radians(15.0)),
        pose_tol=0.25,
        joint_indices=list(range(14)),
    )
    upright = microduck_mdp.backflip_upright_after_roll(env)
    height = microduck_mdp.backflip_height_after_roll(env, 0.115)

    assert composite.tolist() == [1.0, 0.0]
    assert sharp.tolist() == [1.0, 0.0]
    assert success.tolist() == [1.0, 0.0]
    assert upright.tolist() == [1.0, 1.0]
    assert height.tolist() == [1.0, 1.0]


def test_backflip_head_latch_accepts_only_a_grounded_backward_sagittal_transit():
    candidate = microduck_mdp._backflip_head_latch_candidate(
        # Valid, no contact, wrong direction, too early, too late, side roll.
        head_contact=torch.tensor([True, False, True, True, True, True]),
        accum=torch.tensor([
            math.radians(120.0),
            math.radians(120.0),
            math.radians(120.0),
            math.radians(10.0),
            math.radians(270.0),
            math.radians(120.0),
        ]),
        omega_fwd=torch.tensor([2.0, 2.0, -2.0, 2.0, 2.0, 2.0]),
        lateral_axis_z_abs=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.9]),
    )
    assert candidate.tolist() == [True, False, False, False, False, False]
