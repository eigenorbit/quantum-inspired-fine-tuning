"""The split of ResNet152 at layer4.2, what trains, BN, normalisation, spectrum."""

import torch

from experiments.resnet_adapt import backbone

LAYER42_PARAMS = 1_048_576 + 2_359_296 + 1_048_576 + 2 * (512 + 512 + 2048)   # convs + BN affine


def _x(n=2, seed=0):
    return torch.rand(n, 3, 224, 224, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def test_front_plus_classifier_equals_torchvision(net):
    x = _x()
    captured = {}
    hook = net.avgpool.register_forward_hook(lambda m, i, o: captured.update(pool=o))
    try:
        ref_logits = net(backbone.Normalize()(x))
    finally:
        hook.remove()
    front = backbone.Front(net)
    clf = backbone.Classifier(net.layer4[2], 3, seed=0)
    pooled = clf.features(front(x))
    ref_pool = captured["pool"].flatten(1)
    assert torch.allclose(pooled, ref_pool, rtol=1e-5, atol=1e-5 * ref_pool.abs().max())
    logits = net.fc(pooled)
    assert torch.allclose(logits, ref_logits, rtol=1e-5, atol=1e-5 * ref_logits.abs().max())


def test_only_last_block_and_head_train(net):
    front = backbone.Front(net)
    assert not any(p.requires_grad for p in front.parameters())
    clf = backbone.Classifier(net.layer4[2], 5, seed=0)
    names = {n.split(".")[0] for n, p in clf.named_parameters() if p.requires_grad}
    assert names == {"block", "head"}
    assert clf.num_trainable() == LAYER42_PARAMS + 2048 * 5 + 5
    # the pretrained block itself is not touched: the classifier owns a copy
    assert clf.block is not net.layer4[2]


def test_batchnorm_stays_in_eval_and_statistics_frozen(net):
    front = backbone.Front(net)
    front.train()
    assert not front.training and not any(m.training for m in front.modules())
    clf = backbone.Classifier(net.layer4[2], 3, seed=0)
    clf.train()
    bns = [m for m in clf.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and not any(m.training for m in bns)
    stats = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
    opt = torch.optim.Adam([p for p in clf.parameters() if p.requires_grad], lr=1e-2)
    h = torch.rand(4, 2048, 7, 7, generator=torch.Generator().manual_seed(3))
    loss = torch.nn.functional.cross_entropy(clf(h), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    opt.step()
    for m, (mu, var) in zip(bns, stats):
        assert torch.equal(m.running_mean, mu) and torch.equal(m.running_var, var)


def test_normalisation_inside_model_and_pixel_gradient(net):
    x = _x(1).requires_grad_(True)
    norm = backbone.Normalize()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(norm(x), (x - mean) / std)
    front = backbone.Front(net)
    clf = backbone.Classifier(net.layer4[2], 3, seed=0).eval()
    full = backbone.FullModel(front, clf)
    z = full(x)
    z[0, 1].backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    with torch.no_grad():
        assert torch.allclose(z, clf(front(x)))


def test_head_depends_only_on_seed(net):
    a = backbone.Classifier(net.layer4[2], 3, seed=4)
    b = backbone.Classifier(net.layer4[2], 3, seed=4)
    c = backbone.Classifier(net.layer4[2], 3, seed=5)
    assert torch.equal(a.head.linear.weight, b.head.linear.weight)
    assert not torch.equal(a.head.linear.weight, c.head.linear.weight)
    assert torch.allclose(a.head.get_weight().norm(dim=1), torch.ones(3))
    for (n1, p1), (n2, p2) in zip(a.block.state_dict().items(),
                                  net.layer4[2].state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_spectrum_ratio_one_at_start_and_moves_with_training(net):
    clf = backbone.Classifier(net.layer4[2], 3, seed=0)
    s0 = backbone.block_spectrum(clf.block, net.layer4[2])
    assert [r["name"] for r in s0["layers"]] == ["conv1", "conv2", "conv3"]
    assert [r["matrix_shape"] for r in s0["layers"]] == [[512, 2048], [512, 4608], [2048, 512]]
    assert abs(s0["ratio_sigma_max_min"] - 1) < 1e-12 and abs(s0["ratio_sigma_max_max"] - 1) < 1e-12
    clf.train()
    opt = torch.optim.Adam([p for p in clf.parameters() if p.requires_grad], lr=1e-2)
    h = torch.rand(4, 2048, 7, 7, generator=torch.Generator().manual_seed(3))
    torch.nn.functional.cross_entropy(clf(h), torch.tensor([0, 1, 2, 0])).backward()
    opt.step()
    s1 = backbone.block_spectrum(clf.block, net.layer4[2])
    assert max(abs(r["ratio_sigma_max"] - 1) for r in s1["layers"]) > 1e-4
