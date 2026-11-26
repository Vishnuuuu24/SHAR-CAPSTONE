import unittest

import torch
from torch import nn

from core.baseline_rows import ROWS, FrameMeanHead, LSTMSequenceHead, build_head
from models.classifiers import CNNLSTMTemporalBaseline, GlobalAverageLinearHead, SameC5MLPHead


class TinyC5(nn.Module):
    out_channels = 16

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, self.out_channels, kernel_size=3, padding=1)

    def forward(self, images):
        return self.features(images)


class MatchesRegisteredModels(unittest.TestCase):
    def test_lstm_head_on_gap_features_equals_cnn_lstm(self):
        torch.manual_seed(0)
        registered = CNNLSTMTemporalBaseline(TinyC5(), projection_dim=8, hidden_dim=6, num_layers=1, num_classes=14, dropout=0.0).eval()
        head = LSTMSequenceHead(16, 14, projection_dim=8, hidden_dim=6).eval()
        head.projection.load_state_dict(registered.projection.state_dict())
        head.lstm.load_state_dict(registered.lstm.state_dict())
        head.classifier.load_state_dict(registered.classifier.state_dict())
        clips = torch.randn(2, 4, 3, 8, 8)
        lengths = torch.tensor([4, 2])
        with torch.no_grad():
            want = registered(clips, lengths, ["a", "b"])
            gap = registered.pool(registered.backbone(clips.reshape(8, 3, 8, 8))).flatten(1).reshape(2, 4, 16)
            mask = torch.arange(4).unsqueeze(0) >= lengths.unsqueeze(1)
            got = head(gap, key_padding_mask=mask)
        torch.testing.assert_close(got, want)

    def test_trailing_padding_does_not_change_the_lstm_output(self):
        torch.manual_seed(1)
        head = LSTMSequenceHead(5, 3, projection_dim=4, hidden_dim=4).eval()
        x = torch.randn(1, 3, 5)
        padded = torch.cat([x, torch.zeros(1, 5, 5)], 1)
        mask = torch.tensor([[False] * 3 + [True] * 5])
        with torch.no_grad():
            torch.testing.assert_close(head(padded, key_padding_mask=mask), head(x))

    def test_leading_padding_rejected(self):
        head = LSTMSequenceHead(5, 3)
        with self.assertRaises(ValueError):
            head(torch.randn(1, 3, 5), key_padding_mask=torch.tensor([[True, False, False]]))

    def test_frame_heads_equal_registered_gap_heads_per_frame(self):
        torch.manual_seed(2)
        c5 = torch.randn(3, 16, 4, 4)
        gap = c5.mean((2, 3))
        lin = GlobalAverageLinearHead(16, 14).eval()
        mine = build_head("R_lin", 16, 14, 8).eval()
        mine.frame_classifier.load_state_dict(lin.classifier.state_dict())
        mlp = SameC5MLPHead(16, 512, 14, 0.2).eval()
        mine_mlp = build_head("R_mlp", 16, 14, 8).eval()
        mine_mlp.frame_classifier.load_state_dict(mlp.mlp.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(mine(gap.unsqueeze(1)), lin(c5))
            torch.testing.assert_close(mine_mlp(gap.unsqueeze(1)), mlp(c5))

    def test_frame_mean_ignores_padding(self):
        head = FrameMeanHead(nn.Identity())
        seq = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        out = head(seq, key_padding_mask=torch.tensor([[False, False, True]]))
        torch.testing.assert_close(out, torch.tensor([[2.0, 4.0]]))

    def test_rows(self):
        self.assertEqual(ROWS, ("R_lin", "R_mlp", "R_lstm", "D_lstm", "D_step"))


if __name__ == "__main__":
    unittest.main()
