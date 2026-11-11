"""The output names are a contract: finished runs, attack folders and reports
already exist under these names, and resuming or skipping them depends on the
code finding them again. Every name below exists in the published results;
changing one breaks the resume of real runs."""

from pathlib import Path

import pytest

from experiments.common import layout


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMA_REPRO_OUT", str(tmp_path))
    return tmp_path


def rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def test_suffix_rules():
    assert layout.batch_tag(256) == "" and layout.batch_tag(32) == "_bs32"
    assert layout.attack_tag("pgd") == "" and layout.attack_tag("apgd-t") == "_apgd-t"
    assert layout.last_tag("best") == "" and layout.last_tag("final") == "_last"
    assert layout.run_name("lora") == "lora" and layout.run_name("lora", "stable") == "lora_stable"


def test_package_folders(out):
    b, r = layout.BRONET, layout.RESNET
    assert rel(b.backbone_dir(), out) == "bronet_mlp/backbone"
    assert rel(b.features_dir(2), out) == "bronet_mlp/features/step_2"
    assert rel(r.features_dir(1), out) == "resnet_adapt/features/step_1"
    assert rel(b.runs_dir(1), out) == "bronet_mlp/runs/step_1"
    assert rel(b.run_dir(2, "full", 0, "stable", 32), out) == "bronet_mlp/runs/step_2_bs32/full_stable/seed0"
    assert rel(r.run_dir(1, "head-only", 2, "best", 32), out) == "resnet_adapt/runs/step_1_bs32/head-only/seed2"
    assert rel(b.tuning_dir(2, 32), out) == "bronet_mlp/tuning/step_2_bs32"
    assert rel(b.tune_run_dir(1, "lora", 3e-4, 0), out) == "bronet_mlp/tuning/step_1/lora/lr0.0003/seed0"
    assert rel(r.tune_run_dir(2, "full", 1e-3, 0, 32), out) == "resnet_adapt/tuning/step_2_bs32/full/lr0.001/seed0"
    assert rel(b.figures_dir(), out) == "bronet_mlp/results/figures"


def test_lossy_folders(out):
    b = layout.BRONET
    assert rel(b.lossy_dir(2, "best", "native"), out) == "bronet_mlp/results/lossy/step_2_best_native"
    assert rel(b.lossy_features_file(1, "input", "jpeg90"), out) == \
        "bronet_mlp/lossy_features/step_1/input/jpeg90.pt"
    assert rel(b.eps_sweep_run_dir(2, 108, "lora", 1, "stable"), out) == \
        "bronet_mlp/eps_sweep/step_2/eps108/lora_stable/seed1"
    assert rel(b.eps_sweep_run_dir(1, 195, "butterfly", 0, warmup="_wu0.33"), out) == \
        "bronet_mlp/eps_sweep/step_1/eps195_wu0.33/butterfly/seed0"


def test_conv_adapt_folders(out):
    b = layout.BRONET
    assert rel(b.conv_features_dir(2, 1, "float16"), out) == "bronet_mlp/conv_features/step_2/last1_float16"
    assert rel(b.conv_run_dir(1, 2, "butterfly", 0), out) == "bronet_mlp/conv_adapt/step_1/last2/butterfly/seed0"
    assert rel(b.conv_run_dir(2, 1, "lora", 1, "stable"), out) == \
        "bronet_mlp/conv_adapt/step_2/last1/lora_stable/seed1"
    assert rel(b.conv_tuning_dir(2, 1), out) == "bronet_mlp/conv_adapt/tuning/step_2/last1"
    assert rel(b.conv_attack_dir(2, 1, "final"), out) == \
        "bronet_mlp/results/conv_adapt/attacks_step2_last1_final"
    assert rel(b.conv_attack_dir(1, 2, "best", "apgd-t", "mineps"), out) == \
        "bronet_mlp/results/conv_adapt/mineps_step1_last2_best_apgd-t"
    assert rel(b.conv_export_file(2, 1), out) == "bronet_mlp/conv_adapt_results_step2_last1.zip"
    assert rel(b.conv_lossy_dir(2, 1, "final", "native"), out) == \
        "bronet_mlp/results/conv_adapt/lossy_step2_last1_final_native"
    assert rel(b.conv_tune_run_dir(2, 1, "lora", "stable", 0.003, 1e-4), out) == \
        "bronet_mlp/conv_adapt/tuning/step_2/last1/lora_stable/conv0.003_mlp0.0001/seed0"
    assert rel(b.conv_summary_file(2, 1, "final"), out) == \
        "bronet_mlp/results/conv_adapt/summary_step2_last1_last.md"
    assert rel(b.conv_summary_file(2, 1, "best"), out) == \
        "bronet_mlp/results/conv_adapt/summary_step2_last1.md"


def test_warmup_and_loss_tags(out):
    assert layout.warmup_tag() == "" and layout.warmup_tag(0.0, 0.0) == ""
    assert layout.warmup_tag(0.33) == "_wu0.33"
    assert layout.warmup_tag(0.33, 9 / 255) == "_wu0.33from9"
    assert layout.loss_tag(36 / 255) == ""                       # the protocol's own tuning
    assert layout.loss_tag(36 / 255, 0.33) == "_wu0.33"
    assert layout.loss_tag(195 / 255, 0.33) == "_eps195_wu0.33"
    assert layout.loss_tag(195 / 255) == "_eps195"
    b = layout.BRONET
    assert rel(b.tuning_dir(1, 256, "_eps195_wu0.33"), out) == "bronet_mlp/tuning/step_1_eps195_wu0.33"
    assert rel(b.tune_run_dir(1, "butterfly", 0.03, 0, 256, "_eps195_wu0.33"), out) == \
        "bronet_mlp/tuning/step_1_eps195_wu0.33/butterfly/lr0.03/seed0"


def test_attack_folders(out):
    b, r = layout.BRONET, layout.RESNET
    assert rel(b.attack_dir(1, 256, "headline"), out) == "bronet_mlp/results/attacks/step_1_headline"
    assert rel(b.attack_dir(2, 256, "headline", "apgd-t"), out) == \
        "bronet_mlp/results/attacks/step_2_headline_apgd-t"
    assert rel(b.mineps_dir(1, 256, "headline"), out) == "bronet_mlp/results/attacks/step_1_headline_mineps"
    assert rel(b.attack_dir(2, 32, "best", "apgd-t"), out) == "bronet_mlp/results/attacks/step_2_bs32_best_apgd-t"
    assert rel(r.attack_dir(1, 32, "best"), out) == "resnet_adapt/results/attacks/step_1_bs32_best"
    assert rel(r.attack_dir(2, 32, "best", "pgd", "_eps1-2-4-9-18"), out) == \
        "resnet_adapt/results/attacks/step_2_bs32_best_eps1-2-4-9-18"
    assert rel(r.mineps_dir(2, 32, "final", "square"), out) == \
        "resnet_adapt/results/attacks/step_2_bs32_final_mineps_square"


def test_file_names():
    assert layout.checkpoint_file("best") == "best.pt"
    assert layout.epoch_checkpoint_file(7) == "epoch_007.pt"
    assert layout.metrics_file("final") == "metrics_final.json"
    assert layout.predictions_file("best") == "predictions_best.csv"
    assert layout.arrays_file("certificates", "best") == "certificates_best.npz"
    assert layout.arrays_file("outputs", "final") == "outputs_final.npz"
    assert layout.run_attack_csv("pgd", "best") == "pgd_l2_best.csv"
    assert layout.run_attack_csv("apgd-t", "best") == "apgd_t_l2_best.csv"
    assert layout.run_mineps_csv("pgd", "best") == "min_eps_pgd_best.csv"
    assert layout.tuning_summary_file(2, 32) == "tuning_step2_bs32.md"
    assert layout.summary_file(1, 256, "best") == "summary_step1.md"
    assert layout.summary_file(2, 32, "final") == "summary_step2_bs32_last.md"
    assert layout.mcnemar_file(1, 32, "final", certified=True) == "mcnemar_cert_step1_bs32_last.csv"
    assert layout.mcnemar_file(2, 256, "best") == "mcnemar_step2.csv"
    assert layout.adaptation_file(2, 32) == "adaptation_step2_bs32.csv"
    assert layout.runs_table_file(1, 32) == "runs_step1_bs32.csv"
    assert layout.lipschitz_file(2, 32, "md") == "lipschitz_step2_bs32.md"
    assert layout.attack_summary_file(1, "md") == "attacks_step1.md"
    assert layout.attack_summary_file(2, "csv", "min_eps") == "min_eps_step2.csv"


def test_figure_names():
    assert layout.attack_figure(1, 256, "headline", "apgd-t") == "attacks_step1_headline_apgd-t.png"
    assert layout.attack_figure(None, 256, "headline") == "attacks_both_steps_headline.png"
    assert layout.attack_figure(None, 32, "best", "pgd", "_eps1-2-4-9-18") == \
        "attacks_both_steps_bs32_best_eps1-2-4-9-18.png"
    assert layout.certified_figure(1, 32, "rigorous", "final") == "certified_accuracy_step1_bs32_rigorous_last.png"
    assert layout.certified_curves_figure(256, "best", "official") == \
        "certified_curves_both_steps_best_official.png"
    assert layout.lipschitz_growth_figure(256, "best") == "lipschitz_growth_best.png"


def test_package_entry_points_agree(out):
    """The package-level helpers that the rest of the code and the figure scripts
    call give the names of `layout`."""
    from experiments.bronet_mlp import attacks, env, min_eps, tune
    from experiments.resnet_adapt import attacks as rattacks
    from experiments.resnet_adapt import min_eps as rmin
    from experiments.resnet_adapt import paths
    from experiments.resnet_adapt import tune as rtune

    assert env.run_dir(2, "lora", 1, "stable", 32) == layout.BRONET.run_dir(2, "lora", 1, "stable", 32)
    assert paths.run_dir(2, "lora", 1, "stable", 32) == layout.RESNET.run_dir(2, "lora", 1, "stable", 32)
    assert rel(attacks.out_dir(2, "best", 32, "apgd-t"), out) == "bronet_mlp/results/attacks/step_2_bs32_best_apgd-t"
    assert rel(min_eps.out_dir(2, "headline"), out) == "bronet_mlp/results/attacks/step_2_headline_mineps"
    assert rel(rattacks.out_dir(2, 32, "best", (1, 2, 4, 9, 18)), out) == \
        "resnet_adapt/results/attacks/step_2_bs32_best_eps1-2-4-9-18"
    assert rel(rmin.out_dir(1, 32, "best"), out) == "resnet_adapt/results/attacks/step_1_bs32_best_mineps"
    assert rel(tune.split_path(1), out) == "bronet_mlp/tuning/step_1/val_split.json"
    assert rel(tune.selection_path(1, 32), out) == "bronet_mlp/tuning/step_1_bs32/selection.json"
    assert rel(rtune.selection_path(2, 32), out) == "resnet_adapt/tuning/step_2_bs32/selection.json"
    assert attacks.run_csv_name("square", "final") == "square_l2_final.csv"
