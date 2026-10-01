import pytest
from affective_task_transfer.reporting import conditional_task_deltas, architecture_deltas, summarize_architecture_deltas

def test_conditional_pair_to_triple_compares_same_task_seed_and_metric():
    common={"dataset_hash":"data","backbone":"bert-base-uncased","seed":42,"task":"intensity","architecture":"soft_sharing","metric":"mae"}
    pair=common|{"tasks":"emotion+intensity","condition":"soft_emotion_intensity","value":0.40}
    triple=common|{"tasks":"sentiment+emotion+intensity","condition":"soft_sentiment_emotion_intensity","value":0.35}
    unrelated=pair|{"seed":52,"value":0.10}
    result=conditional_task_deltas([pair,triple,unrelated])
    assert len(result)==1
    assert result[0]["added_task"]=="sentiment"
    assert result[0]["delta_triple_minus_pair"]==pytest.approx(-0.05)
    assert result[0]["benefit"]==pytest.approx(0.05)


def test_architecture_contrast_is_seed_and_task_matched():
    common={"dataset_hash":"data","backbone":"bert-base-uncased","tasks":"emotion+intensity",
            "seed":42,"task":"intensity","metric":"emotion_macro_f1"}
    rows=[common|{"architecture":"hard_sharing","condition":"hard_sharing_emotion_intensity","value":0.30},
          common|{"architecture":"soft_sharing","condition":"soft_sharing_emotion_intensity","value":0.35},
          common|{"architecture":"mmoe","condition":"mmoe_emotion_intensity","value":0.32},
          common|{"architecture":"soft_sharing","condition":"ablation_projection0_l20","value":0.50},
          common|{"architecture":"matched_stl","condition":"stl_intensity","value":0.20},
          common|{"architecture":"adapters","condition":"adapters_emotion_intensity","seed":52,"value":0.40}]
    deltas=architecture_deltas(rows)
    assert len(deltas)==3
    soft_hard=next(r for r in deltas if {r["architecture_a"],r["architecture_b"]}=={"hard_sharing","soft_sharing"})
    assert soft_hard["benefit_a_over_b"]==pytest.approx(-0.05)
    summary=summarize_architecture_deltas(deltas)
    assert len(summary)==3 and all(r["n_paired_seeds"]==1 for r in summary)
