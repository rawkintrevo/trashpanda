from trashpanda.config import get_recipe, load_config, resolve_training_config


def test_fixed_recipes():
    tiny = get_recipe("tiny")
    assert (tiny.model.vocab_size, tiny.model.num_layers, tiny.model.hidden_size) == (32, 2, 32)
    assert tiny.model.context_length == 32
    assert (tiny.training.batch_size, tiny.training.microbatch_size) == (4, 2)
    assert tiny.training.token_budget == 744
    assert tiny.training.checkpoint_interval == 3
    assert tiny.training.precision == "fp32"
    production = get_recipe("300m-a100")
    assert production.parameter_count == 298_103_808
    assert production.training.token_budget == 6_000_000_000


def test_strict_flat_overrides_and_precedence():
    config = load_config("tiny", overrides={"token_budget": 248, "repeat_data": False})
    assert config.training.token_budget == 248
    assert config.training.repeat_data is False
    config = resolve_training_config(config, microbatch_size=1, learning_rate=1e-4)
    assert config.training.microbatch_size == 1
    assert config.training.learning_rate == 1e-4
    try:
        load_config("tiny", overrides={"training": {"token_budget": 1}})
    except ValueError as exc:
        assert "only allow" in str(exc)
    else:
        raise AssertionError("nested or unknown config keys must be rejected")
