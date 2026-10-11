import json
from pathlib import Path
import unittest

from backend.climb_path.tokens import (
    EOS_TOKEN,
    MIDDLE_TOKEN,
    additional_special_tokens,
    encode_route,
    hold_label,
    hold_token,
    prompt_tokens,
)
from backend.evaluation.evaluate_structured_model import (
    allowed_completion_token_ids,
    route_fingerprint,
)
from backend.training.train_structured import (
    CompletionOnlyCollator,
    grade_sampling_weights,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "data_contract" / "v1"


class Phase2TokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board_holds = json.loads(
            (CONTRACT_DIR / "board_holds.json").read_text(encoding="utf-8")
        )["holds"]

    def test_atomic_vocabulary_is_unique_and_complete(self):
        tokens = additional_special_tokens(self.board_holds)
        self.assertEqual(len(tokens), 3 + 9 + 141)
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertEqual(hold_label(hold_token("F10")), "F10")

    def test_multi_endpoint_prompt_and_completion_only_mask(self):
        record = {
            "grade": "7A",
            "start_holds": ["A4", "F4"],
            "middle_holds": ["D8", "G12"],
            "end_holds": ["E17", "G17"],
        }
        tokens = prompt_tokens(
            grade=record["grade"],
            start_holds=record["start_holds"],
            end_holds=record["end_holds"],
        )
        all_tokens = tokens + [hold_token(label) for label in record["middle_holds"]] + [
            EOS_TOKEN
        ]
        token_ids = {token: index + 1 for index, token in enumerate(all_tokens)}
        encoding = encode_route(record, token_ids.__getitem__, max_length=32)

        self.assertEqual(tokens[-1], MIDDLE_TOKEN)
        self.assertEqual(encoding.labels[: encoding.prompt_length], (-100,) * len(tokens))
        self.assertEqual(
            encoding.labels[encoding.prompt_length :],
            tuple(token_ids[token] for token in all_tokens[len(tokens) :]),
        )

    def test_dynamic_collator_masks_padding(self):
        collator = CompletionOnlyCollator(pad_token_id=99)
        batch = collator(
            [
                {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
                {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
            ]
        )
        self.assertEqual(batch["input_ids"].tolist()[1], [4, 5, 99])
        self.assertEqual(batch["attention_mask"].tolist()[1], [1, 1, 0])
        self.assertEqual(batch["labels"].tolist()[1], [-100, 5, -100])

    def test_evaluation_fingerprint_matches_contract(self):
        first = json.loads(
            (CONTRACT_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertEqual(
            route_fingerprint(
                first["start_holds"], first["middle_holds"], first["end_holds"]
            ),
            first["fingerprint"],
        )

    def test_decoder_excludes_prompt_and_generated_holds(self):
        allowed = allowed_completion_token_ids(
            current_ids=[10, 20, 30, 20],
            hold_ids={20, 30, 40},
            eos_id=99,
            prompt_length=3,
            minimum_middle_holds=2,
        )
        self.assertEqual(allowed, [40])
        allowed_after_minimum = allowed_completion_token_ids(
            current_ids=[10, 20, 30, 20, 40],
            hold_ids={20, 30, 40, 50},
            eos_id=99,
            prompt_length=3,
            minimum_middle_holds=2,
        )
        self.assertEqual(allowed_after_minimum, [50, 99])

    def test_grade_sampling_weights_correct_imbalance_without_growing_epoch(self):
        records = [{"grade": "6B+"}] * 4 + [{"grade": "7A"}]

        self.assertIsNone(grade_sampling_weights(records, 0.0))
        weights = grade_sampling_weights(records, 1.0)

        self.assertIsNotNone(weights)
        self.assertEqual(len(weights), len(records))
        self.assertEqual(weights[:4], [0.25] * 4)
        self.assertEqual(weights[4], 1.0)


if __name__ == "__main__":
    unittest.main()
