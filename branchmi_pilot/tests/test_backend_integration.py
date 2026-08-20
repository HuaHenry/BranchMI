import json
import math

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from branchmi_pilot.backend import SamplingOptions, TransformersBackend
from branchmi_pilot.config import (
    AnalysisConfig,
    CheckpointConfig,
    DatasetConfig,
    ExperimentConfig,
    GenerationConfig,
    ModelConfig,
    PromptConfig,
)
from branchmi_pilot.experiment import PilotExperiment


def test_tiny_local_transformers_backend(tmp_path):
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "Solve": 3,
        "the": 4,
        "problem": 5,
        "answer": 6,
        "is": 7,
        "one": 8,
        "two": 9,
        "three": 10,
    }
    tokenizer_object = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_object.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(tmp_path)

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=1,
            n_head=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model.generation_config.suppress_tokens = [0, 1, 2]
    model.save_pretrained(tmp_path)

    backend = TransformersBackend(
        ModelConfig(
            name_or_path=str(tmp_path),
            dtype="float32",
            device="cpu",
            device_map=None,
            attn_implementation=None,
            trust_remote_code=False,
            max_context_tokens=64,
        )
    )
    prompt = tokenizer.encode("Solve the problem", add_special_tokens=False)
    generated = backend.generate(
        [prompt], SamplingOptions(max_new_tokens=4, do_sample=False)
    )
    statistics = backend.next_token_statistics(prompt, top_k=2)
    candidates = statistics["candidates"]
    lookahead = backend.generate_lookahead(
        [prompt + [candidate["token_id"]] for candidate in candidates],
        max_new_tokens=3,
        weights=[candidate["branch_weight"] for candidate in candidates],
    )

    assert len(generated.token_ids) == 1
    assert len(candidates) == 2
    assert statistics["entropy_nats"] > 0
    assert math.isfinite(lookahead.mean_js_nats)
    assert lookahead.mean_js_nats >= 0


def test_tiny_local_end_to_end_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    model_path = tmp_path / "model"
    model_path.mkdir()
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "Solve": 3,
        "one": 4,
        "plus": 5,
        "answer": 6,
        "two": 7,
        "three": 8,
    }
    tokenizer_object = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_object.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(model_path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=1,
            n_head=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model.generation_config.suppress_tokens = [0, 1, 2]
    model.save_pretrained(model_path)

    dataset_path = tmp_path / "problems.jsonl"
    dataset_path.write_text(
        json.dumps({"id": "tiny-1", "problem": "one plus one", "answer": "two"}) + "\n",
        encoding="utf-8",
    )
    cfg = ExperimentConfig(
        run_name="e2e",
        output_dir=str(tmp_path / "outputs"),
        save_full_text=False,
        dataset=DatasetConfig(
            kind="jsonl",
            path=str(dataset_path),
            id_field="id",
            question_field="problem",
            answer_field="answer",
            limit=1,
            shuffle=False,
        ),
        model=ModelConfig(
            name_or_path=str(model_path),
            dtype="float32",
            device="cpu",
            device_map=None,
            attn_implementation=None,
            trust_remote_code=False,
            max_context_tokens=64,
        ),
        prompt=PromptConfig(
            system="",
            user_template="Solve {question}",
            probe="answer",
            chat_template_kwargs={},
            probe_chat_template_kwargs={},
        ),
        generation=GenerationConfig(
            baseline_max_new_tokens=12,
            oracle_max_total_tokens=12,
            candidate_top_k=2,
            lookahead_tokens=2,
            probe_samples=2,
            probe_max_new_tokens=2,
        ),
        checkpoints=CheckpointConfig(
            strategy="uniform",
            max_per_problem=1,
            min_generated_tokens=2,
            tail_exclusion_tokens=2,
        ),
        analysis=AnalysisConfig(bootstrap_samples=2),
    )

    run_dir = PilotExperiment(cfg).run()

    assert (run_dir / "problems.jsonl").exists()
    assert (run_dir / "checkpoints.jsonl").exists()
    assert (run_dir / "summary.json").exists()
