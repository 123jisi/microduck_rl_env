import math
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
)


def test_cfg_has_postroll_stability_rewards():
    cfg = make_microduck_roulade_env_cfg()
    for name in (
        "roulade_recovery_composite",
        "roulade_landing_composite",
        "roulade_landing_sharp",
        "roulade_stand_tax",
        "roulade_head_contact_after_roll",
        "roulade_settled_standing",
        "roulade_standing_success",
        "roulade_intermediate_standing",
    ):
        assert name in cfg.rewards, name


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
    assert cfg.rewards["roulade_upright_after_roll"].func is microduck_mdp.roulade_upright_after_roll
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
        lambda _env, _lo, _hi: torch.ones(2),
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
        lambda _env, _lo, _hi: torch.ones(2),
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
