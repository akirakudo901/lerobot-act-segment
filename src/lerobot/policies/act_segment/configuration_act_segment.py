#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

from dataclasses import dataclass

from hybrid_eval.segment.configuration_segment import (
    SegmentPolicyConfigMixin,
    rewrite_legacy_ompl_cli_args,
)

from lerobot.configs import PreTrainedConfig

from ..act.configuration_act import ACTConfig
from ..act.configuration_hybrid_act import ACTHybridConfigMixin


@PreTrainedConfig.register_subclass("act_segment")
@dataclass
class ACTSegmentConfig(ACTConfig, SegmentPolicyConfigMixin, ACTHybridConfigMixin):
    """ACT with an auxiliary per-chunk-step MP/L BIO label head.

    Hybrid rollout + segment CE knobs come from :class:`SegmentPolicyConfigMixin`.
    Action L1 / preprocessor knobs come from :class:`ACTHybridConfigMixin`.
    """

def _patch_pretrained_ompl_cli_overrides() -> None:
    """Rewrite legacy ``--ompl_*`` CLI overrides when loading a pretrained policy."""
    orig = PreTrainedConfig.from_pretrained
    if getattr(orig, "_ompl_cli_rewrite", False):
        return
    orig_func = orig.__func__

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **policy_kwargs):  # type: ignore[no-untyped-def]
        overrides = policy_kwargs.get("cli_overrides")
        if overrides:
            policy_kwargs["cli_overrides"] = rewrite_legacy_ompl_cli_args(list(overrides))
        return orig_func(cls, pretrained_name_or_path, **policy_kwargs)

    from_pretrained._ompl_cli_rewrite = True  # type: ignore[attr-defined]
    PreTrainedConfig.from_pretrained = from_pretrained


_patch_pretrained_ompl_cli_overrides()
