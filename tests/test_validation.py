import numpy as np
import pandas as pd

from wind_bayes_twin.experiments import log_episodes
from wind_bayes_twin.model import ALARM_GROUPS, FAULTS, ModelSpec
from wind_bayes_twin.validation import spec_at, twin_alarms


def probs(rows):
    p = np.zeros((len(rows), len(FAULTS)))
    for d, row in enumerate(rows):
        for f, v in row.items():
            p[d, FAULTS.index(f)] = v
    return p


def test_group_alarm_and_rearm():
    # efficiency and derating share the power group: 0.5 + 0.45 > 0.9 raises one alarm, diagnosed as efficiency
    p = probs([{}, {"eff": 0.5, "derate": 0.45}, {"eff": 0.95}, {"eff": 0.3}, {"eff": 0.95}])
    alarms, armed = twin_alarms(p, np.ones(len(ALARM_GROUPS), dtype=bool), ModelSpec())
    assert alarms == [(1, FAULTS.index("eff")), (4, FAULTS.index("eff"))]
    assert not armed[0] and armed[1:].all()


def test_not_armed_means_no_new_alarm():
    p = probs([{"yaw": 0.95}, {"yaw": 0.97}])
    armed = np.ones(len(ALARM_GROUPS), dtype=bool)
    armed[1] = False  # the yaw group already alarmed before
    alarms, _ = twin_alarms(p, armed, ModelSpec())
    assert alarms == []


def test_threshold_spec():
    s = spec_at(0.7)
    assert s.p_alarm == 0.7 and np.isclose(s.p_rearm, 0.3)
    assert np.isclose(spec_at(0.5).p_rearm, 0.1)


def test_log_episodes_merge_within_two_weeks():
    t = pd.to_datetime(["2020-01-01", "2020-01-05", "2020-03-01", "2020-01-02"]).tz_localize("UTC")
    status = pd.DataFrame({"turbine": [1, 1, 1, 2], "start": t, "end": t,
                           "message": ["Anemometer defect", "Icing (anemometer)", "Anemometer defect", "Vane 2 defect"]})
    eps = log_episodes(status, 2019, 2021)
    anemo = eps[eps["fault"] == "anemo"]
    assert len(anemo) == 2 and anemo.iloc[0]["messages"] == "Anemometer defect; Icing (anemometer)"
    assert eps[eps["fault"] == "yaw"]["turbine"].tolist() == [2]
