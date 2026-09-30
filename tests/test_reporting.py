import pytest
from affective_task_transfer.reporting import conditional_task_deltas

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
