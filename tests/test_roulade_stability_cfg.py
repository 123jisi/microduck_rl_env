import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    POSTROLL_BALANCE_GRACE_S,
    POSTROLL_INITIAL_SETTLE_SCALE,
    POSTROLL_RECOVERY_PROB,
    POSTROLL_SETTLE_RAMP_S,
    POSTROLL_STEP_WINDOW_S,
    make_microduck_roulade_env_cfg,
)


def test_cfg_has_postroll_stability_rewards():
    cfg = make_microduck_roulade_env_cfg()
    for name in (
        "roulade_recovery_composite",
        "roulade_recovery_step",
        "roulade_landing_composite",
        "roulade_natural_stance",
        "roulade_landing_sharp",
        "roulade_head_latch",
        "roulade_stand_tax",
        "roulade_head_contact_after_roll",
        "roulade_settled_standing",
        "roulade_standing_success",
        "roulade_intermediate_standing",
    ):
        assert name in cfg.rewards, name


def test_head_top_alignment_is_continuous_and_strongly_rejects_face_plant():
    score = microduck_mdp._head_top_alignment_score(
        torch.tensor([0.6, 0.0, -0.3, -1.0])
    )
    assert torch.all(score[1:] > score[:-1])
    assert torch.allclose(score, torch.tensor([0.04, 0.25, 0.4225, 1.0]))


def test_reverse_curriculum_only_prelatches_spawns_past_head_window():
    is_mid = torch.tensor([False, True, True, True])
    pitch = torch.tensor(
        [0.0, math.radians(90.0), math.radians(169.0), math.radians(171.0)]
    )
    prelatched = microduck_mdp._roulade_spawn_head_latch(is_mid, pitch)
    assert prelatched.tolist() == [False, False, False, True]


def test_reverse_curriculum_has_a_distinct_postroll_recovery_bucket():
    standing, midroll, recovery = microduck_mdp._roulade_spawn_masks(
        torch.tensor([0.10, 0.50, 0.95]),
        standing_prob=0.45,
        midroll_prob=0.45,
        recovery_prob=0.10,
    )
    assert standing.tolist() == [True, False, False]
    assert midroll.tolist() == [False, True, False]
    assert recovery.tolist() == [False, False, True]


def test_head_latch_bonus_is_one_shot_and_rate_normalized(monkeypatch):
    asset = SimpleNamespace()

    class Scene:
        def __getitem__(self, _name):
            return asset

    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        scene=Scene(),
        step_dt=0.02,
        _roulade_accum=torch.zeros(2),
        _roulade_max=torch.zeros(2),
        _roulade_paid=torch.zeros(2),
        _roulade_head_latch=torch.tensor([True, True]),
        # Env 1 represents a synthetic late-spawn latch, already paid.
        _roulade_head_latch_paid=torch.tensor([False, True]),
    )
    monkeypatch.setattr(microduck_mdp, "_update_roulade_accum", lambda _env, _asset: None)

    first = microduck_mdp.roulade_head_latch_bonus(env)
    second = microduck_mdp.roulade_head_latch_bonus(env)
    assert first.tolist() == [50.0, 0.0]
    assert second.tolist() == [0.0, 0.0]


def test_precompletion_polish_terms_are_structurally_gated(monkeypatch):
    asset = SimpleNamespace()

    class Scene:
        def __getitem__(self, _name):
            return asset

    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), scene=Scene())
    monkeypatch.setattr(microduck_mdp, "_update_roulade_accum", lambda _env, _asset: None)
    monkeypatch.setattr(
        microduck_mdp,
        "_roulade_completion_gate",
        lambda *_args, **_kwargs: torch.tensor([0.0, 1.0]),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "body_ang_vel_at_height",
        lambda *_args, **_kwargs: torch.tensor([3.0, 3.0]),
    )
    result = microduck_mdp.roulade_arrival_damping(
        env, height_low=0.09, height_high=0.11
    )
    assert result.tolist() == [0.0, 3.0]


def test_cfg_uses_authentic_latch_bridge_and_completion_gated_polish():
    cfg = make_microduck_roulade_env_cfg()
    assert cfg.rewards["roulade_head_pivot"].weight == 1.0
    assert cfg.rewards["roulade_head_latch"].weight == 2.0
    assert cfg.rewards["roulade_head_latch"].func is microduck_mdp.roulade_head_latch_bonus
    assert cfg.rewards["arrival_damping"].func is microduck_mdp.roulade_arrival_damping
    assert (
        cfg.rewards["joint_torque_rate_l2"].func
        is microduck_mdp.roulade_joint_torque_rate_l2
    )
    assert (
        cfg.rewards["gentle_landing"].func
        is microduck_mdp.roulade_gentle_landing_penalty
    )
    spawn = cfg.events["set_roulade_state"].params
    assert spawn["midroll_pitch_power"] == 2.0
    assert spawn["recovery_prob"] == POSTROLL_RECOVERY_PROB
    assert spawn["recovery_forward_speed_range"][0] > 0.0


def test_action_rate_tightening_waits_until_after_discovery_horizon():
    cfg = make_microduck_roulade_env_cfg()
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert [stage["step"] for stage in stages] == [0, 6500 * 24, 8500 * 24]
    assert [stage["weight"] for stage in stages] == [-0.1, -0.2, -0.3]


def test_spawn_curriculum_preserves_recovery_practice_without_dominating():
    cfg = make_microduck_roulade_env_cfg()
    stages = cfg.curriculum["roulade_spawn_mix"].params["param_stages"]
    for stage in stages:
        params = stage["params"]
        assert params["recovery_prob"] == POSTROLL_RECOVERY_PROB
        assert math.isclose(
            params["standing_prob"] + params["midroll_prob"] + params["recovery_prob"],
            1.0,
        )
    assert stages[-1]["params"]["standing_prob"] > stages[-1]["params"]["midroll_prob"]


def test_clean_stand_gate_requires_both_feet_and_current_head_clear():
    gate = microduck_mdp._roulade_clean_stand_gate_from_contacts(
        completion_gate=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        feet_contact=torch.tensor(
            [
                [True, True],
                [True, False],
                [True, True],
                [True, True],
            ]
        ),
        head_contact=torch.tensor([[False], [False], [True], [False]]),
    )
    assert gate.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_final_standing_waits_for_full_roll_then_ramps_after_balance_grace():
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        step_dt=0.02,
        common_step_counter=100,
        _roulade_accum=torch.zeros(3),
        _roulade_max=torch.tensor(
            [math.radians(329.0), math.radians(330.0), math.radians(340.0)]
        ),
        _roulade_paid=torch.zeros(3),
        _roulade_head_latch=torch.tensor([True, True, False]),
        _roulade_head_latch_paid=torch.tensor([True, True, False]),
        _roulade_completion_step=torch.full((3,), -1, dtype=torch.long),
    )

    at_completion = microduck_mdp._roulade_post_completion_settle_scale(
        env,
        completion_angle=math.radians(330.0),
        balance_grace_s=POSTROLL_BALANCE_GRACE_S,
        settle_ramp_s=POSTROLL_SETTLE_RAMP_S,
        initial_scale=POSTROLL_INITIAL_SETTLE_SCALE,
    )
    assert torch.allclose(at_completion, torch.tensor([0.0, 0.0, 0.0]))

    env.common_step_counter = 160  # 1.2 s after full completion.
    after_ramp = microduck_mdp._roulade_post_completion_settle_scale(
        env,
        completion_angle=math.radians(330.0),
        balance_grace_s=POSTROLL_BALANCE_GRACE_S,
        settle_ramp_s=POSTROLL_SETTLE_RAMP_S,
        initial_scale=POSTROLL_INITIAL_SETTLE_SCALE,
    )
    assert torch.allclose(after_ramp, torch.tensor([0.0, 1.0, 0.0]))


def test_all_clean_standing_rewards_share_the_balance_window():
    cfg = make_microduck_roulade_env_cfg()
    for name in (
        "roulade_landing_composite",
        "roulade_natural_stance",
        "roulade_landing_sharp",
        "roulade_settled_standing",
        "roulade_standing_success",
        "roulade_intermediate_standing",
    ):
        params = cfg.rewards[name].params
        assert params["balance_grace_s"] == POSTROLL_BALANCE_GRACE_S
        assert params["settle_ramp_s"] == POSTROLL_SETTLE_RAMP_S
        assert params["initial_settle_scale"] == POSTROLL_INITIAL_SETTLE_SCALE


def test_final_pose_and_stability_terms_include_neck():
    cfg = make_microduck_roulade_env_cfg()
    expected_joints = list(range(14))
    for name in (
        "roulade_recovery_composite",
        "roulade_landing_composite",
        "roulade_settled_standing",
        "roulade_standing_success",
        "roulade_intermediate_standing",
    ):
        assert sorted(cfg.rewards[name].params["joint_indices"]) == expected_joints

    settled = cfg.rewards["roulade_settled_standing"]
    assert settled.params["neck_joint_indices"] == [5, 6, 7, 8]
    assert settled.params["planar_speed_std"] == 0.30
    assert settled.params["trunk_ang_vel_std"] == 1.0
    assert settled.params["neck_speed_std"] == 3.0

    success = cfg.rewards["roulade_standing_success"]
    assert success.params["height_tol"] == 0.0075
    assert success.params["upright_threshold"] == math.cos(math.radians(15.0))
    assert success.params["pose_tol"] == 0.25
    assert success.params["max_planar_speed"] == 0.08
    assert success.params["max_trunk_ang_vel"] == 0.35
    assert success.params["max_neck_speed"] == 0.75


def test_landing_terms_use_clean_support_but_recovery_terms_stay_broad():
    cfg = make_microduck_roulade_env_cfg()
    assert (
        cfg.rewards["roulade_recovery_composite"].func
        is microduck_mdp.roulade_recovery_composite
    )
    assert cfg.rewards["roulade_landing_composite"].func is microduck_mdp.roulade_landing_composite
    assert cfg.rewards["roulade_landing_sharp"].func is microduck_mdp.roulade_landing_sharp
    assert (
        cfg.rewards["roulade_upright_after_roll"].func
        is microduck_mdp.roulade_upright_after_roll
    )
    assert cfg.rewards["roulade_height_after_roll"].func is microduck_mdp.roulade_height_after_roll
    assert cfg.rewards["roulade_upright_after_roll"].weight == 1.5
    assert cfg.rewards["roulade_height_after_roll"].weight == 1.0


def test_sharp_layer_and_normalized_stand_tax_are_strict():
    cfg = make_microduck_roulade_env_cfg()
    sharp = cfg.rewards["roulade_landing_sharp"]
    assert sharp.params["height_std"] == 0.010
    assert sharp.params["upright_std"] == 0.20
    tax = cfg.rewards["roulade_stand_tax"]
    assert tax.weight == 0.25
    assert tax.params["shortfall_scale"] == 0.02
    shortfall = microduck_mdp._normalized_height_shortfall(
        z=torch.tensor([0.115, 0.105, 0.095, 0.075]),
        target_height=0.115,
        shortfall_scale=0.02,
    )
    assert torch.allclose(shortfall, torch.tensor([0.0, 0.5, 1.0, 1.0]))


def test_penalty_signs_match_function_conventions():
    cfg = make_microduck_roulade_env_cfg()
    assert cfg.rewards["roulade_stand_tax"].weight > 0.0
    assert cfg.rewards["roulade_head_contact_after_roll"].weight < 0.0


def test_stability_costs_ramp_only_after_skill_discovery():
    cfg = make_microduck_roulade_env_cfg()
    tax_stages = cfg.curriculum["roulade_stand_tax_weight"].params["weight_stages"]
    head_stages = cfg.curriculum["roulade_head_contact_weight"].params["weight_stages"]
    assert [stage["step"] for stage in tax_stages] == [
        0,
        4000 * 24,
        6500 * 24,
        8500 * 24,
    ]
    assert [stage["weight"] for stage in tax_stages] == [0.25, 0.5, 1.0, 2.0]
    assert [stage["weight"] for stage in head_stages] == [-0.25, -0.5, -1.0, -2.0]


def test_intermediate_success_is_reachable_but_strict_success_is_more_valuable():
    cfg = make_microduck_roulade_env_cfg()
    intermediate = cfg.rewards["roulade_intermediate_standing"]
    strict = cfg.rewards["roulade_standing_success"]
    assert intermediate.func is microduck_mdp.roulade_standing_success_bonus
    assert intermediate.params["height_tol"] == 0.015
    assert intermediate.params["upright_threshold"] == math.cos(math.radians(25.0))
    assert intermediate.params["pose_tol"] == 0.60
    assert strict.weight > intermediate.weight


def test_head_contact_cost_is_gated_by_completion(monkeypatch):
    class Scene:
        def __init__(self):
            self.sensors = {
                "head_ground_contact": SimpleNamespace(
                    data=SimpleNamespace(found=torch.tensor([[1], [1], [0]]))
                )
            }

        def __getitem__(self, _name):
            return SimpleNamespace()

    env = SimpleNamespace(num_envs=3, device=torch.device("cpu"), scene=Scene())
    monkeypatch.setattr(microduck_mdp, "_update_roulade_accum", lambda _env, _asset: None)
    monkeypatch.setattr(
        microduck_mdp,
        "_roulade_completion_gate",
        lambda _env, _lo, _hi, require_head: torch.tensor([1.0, 0.0, 1.0]),
    )
    result = microduck_mdp.roulade_head_contact_after_roll_penalty(env)
    assert result.tolist() == [1.0, 0.0, 0.0]


def test_recovery_bridge_does_not_require_clean_support(monkeypatch):
    asset = SimpleNamespace()

    class Scene:
        def __getitem__(self, _name):
            return asset

    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), scene=Scene())
    monkeypatch.setattr(microduck_mdp, "_update_roulade_accum", lambda _env, _asset: None)
    monkeypatch.setattr(
        microduck_mdp,
        "_roulade_completion_gate",
        lambda _env, _lo, _hi, require_head: torch.tensor([1.0, 0.5]),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_roulade_clean_stand_gate",
        lambda *_args, **_kwargs: torch.zeros(2),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "standing_composite_score",
        lambda *args, **kwargs: torch.tensor([0.4, 0.2]),
    )

    result = microduck_mdp.roulade_recovery_composite(
        env,
        target_height=0.115,
        height_std=0.04,
        upright_std=0.40,
        pose_std=0.40,
        joint_indices=list(range(14)),
    )
    assert torch.allclose(result, torch.tensor([0.4, 0.1]))


def test_recovery_step_reward_needs_imbalance_single_support_and_short_window(monkeypatch):
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_w=torch.tensor(
                [[0.20, 0.0, 0.0], [0.0, 0.0, 0.0], [0.20, 0.0, 0.0]]
            ),
            root_link_ang_vel_w=torch.zeros(3, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
        )
    )

    class Scene:
        def __init__(self):
            self.sensors = {
                "feet_ground_contact": SimpleNamespace(
                    data=SimpleNamespace(found=torch.tensor([[1, 0], [1, 0], [1, 0]]))
                ),
                "head_ground_contact": SimpleNamespace(
                    data=SimpleNamespace(found=torch.zeros(3, 1, dtype=torch.bool))
                ),
            }

        def __getitem__(self, _name):
            return asset

    env = SimpleNamespace(num_envs=3, device=torch.device("cpu"), scene=Scene())
    monkeypatch.setattr(microduck_mdp, "_update_roulade_accum", lambda *_args: None)
    monkeypatch.setattr(
        microduck_mdp,
        "_roulade_post_completion_age_s",
        lambda *_args: torch.tensor([0.20, 0.20, POSTROLL_STEP_WINDOW_S + 0.01]),
    )
    monkeypatch.setattr(microduck_mdp, "_servo_joint_pos", lambda *_args: torch.zeros(3, 14))
    monkeypatch.setattr(
        microduck_mdp, "_servo_default_joint_pos", lambda *_args: torch.zeros(3, 14)
    )

    reward = microduck_mdp.roulade_recovery_step_reward(
        env,
        joint_indices=[0, 1, 2, 3, 4, 9, 10, 11, 12, 13],
        max_age_s=POSTROLL_STEP_WINDOW_S,
        planar_speed_threshold=0.08,
        planar_speed_width=0.16,
        trunk_ang_vel_threshold=0.35,
        trunk_ang_vel_width=0.70,
        tilt_threshold=math.radians(8.0),
        tilt_width=math.radians(12.0),
        pose_error_threshold=0.18,
        pose_error_width=0.25,
    )
    assert reward[0] > 0.0
    assert reward[1] == 0.0
    assert reward[2] == 0.0


def test_natural_stance_score_has_separate_leg_and_neck_gradients(monkeypatch):
    asset = SimpleNamespace()

    class Scene:
        def __getitem__(self, _name):
            return asset

    env = SimpleNamespace(num_envs=3, device=torch.device("cpu"), scene=Scene())
    joint_pos = torch.zeros(3, 14)
    joint_pos[1, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]] = 0.18
    joint_pos[2, [5, 6, 7, 8]] = 0.25
    monkeypatch.setattr(microduck_mdp, "_update_roulade_accum", lambda *_args: None)
    monkeypatch.setattr(microduck_mdp, "_servo_joint_pos", lambda *_args: joint_pos)
    monkeypatch.setattr(
        microduck_mdp, "_servo_default_joint_pos", lambda *_args: torch.zeros(3, 14)
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_roulade_clean_stand_gate",
        lambda *_args, **_kwargs: torch.ones(3),
    )

    score = microduck_mdp.roulade_natural_stance_score(
        env,
        leg_joint_indices=[0, 1, 2, 3, 4, 9, 10, 11, 12, 13],
        neck_joint_indices=[5, 6, 7, 8],
        leg_pose_std=0.18,
        neck_pose_std=0.25,
    )
    expected = torch.tensor([1.0, math.exp(-1.0), math.exp(-1.0)])
    assert torch.allclose(score, expected)


def test_settled_score_has_gradient_from_current_wobbling_policy(monkeypatch):
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_w=torch.tensor([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0]]),
            root_link_ang_vel_w=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
    )

    class Scene:
        def __getitem__(self, _name):
            return asset

    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), scene=Scene())
    monkeypatch.setattr(microduck_mdp, "_update_roulade_accum", lambda _env, _asset: None)
    monkeypatch.setattr(
        microduck_mdp,
        "standing_composite_score",
        lambda *args, **kwargs: torch.ones(2),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_roulade_clean_stand_gate",
        lambda *_args, **_kwargs: torch.ones(2),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_servo_joint_vel",
        lambda _env, _asset: torch.tensor([[0.0] * 14, [3.0] * 14]),
    )
    score = microduck_mdp.roulade_settled_standing_score(
        env,
        target_height=0.115,
        height_std=0.02,
        upright_std=0.25,
        pose_std=0.30,
        joint_indices=list(range(14)),
        neck_joint_indices=[5, 6, 7, 8],
        planar_speed_std=0.30,
        trunk_ang_vel_std=1.0,
        neck_speed_std=3.0,
    )
    assert score[0] == 1.0
    assert 0.0 < score[1] < 0.1


def test_strict_success_rejects_a_fast_two_foot_pass_through(monkeypatch):
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_w=torch.tensor([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0]]),
            root_link_ang_vel_w=torch.zeros(2, 3),
        )
    )

    class Scene:
        def __getitem__(self, _name):
            return asset

    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), scene=Scene())
    monkeypatch.setattr(microduck_mdp, "_update_roulade_accum", lambda _env, _asset: None)
    monkeypatch.setattr(
        microduck_mdp,
        "standing_success_bonus",
        lambda *args, **kwargs: torch.ones(2),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_roulade_clean_stand_gate",
        lambda *_args, **_kwargs: torch.ones(2),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_servo_joint_vel",
        lambda _env, _asset: torch.zeros(2, 14),
    )
    success = microduck_mdp.roulade_standing_success_bonus(
        env,
        target_height=0.115,
        height_tol=0.0075,
        upright_threshold=math.cos(math.radians(15.0)),
        pose_tol=0.25,
        joint_indices=list(range(14)),
        neck_joint_indices=[5, 6, 7, 8],
        max_planar_speed=0.08,
        max_trunk_ang_vel=0.35,
        max_neck_speed=0.75,
    )
    assert success.tolist() == [1.0, 0.0]
