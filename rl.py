import sys
sys.path.append("./btgym")
from btgym import BTgymEnv, BTgymDataset
from fetcher import Fetcher
import cv2



dataset = BTgymDataset(filename='nvda_dataset.csv', parsing_params={"header": 0, "parse_dates": True, "index_col": 0, "sep": ",", "datetime": 0, "open": 1, "high": 2, "low": 3, "close": 4, "volume": 5, "openinterest": -1, "names": [
                       "open", "high", "low", "close", "volume", "exdividend", "splitratio", "adjopen", "adjhigh", "adjlow", "adjclose", "adjvolume"], "timeframe": 1}, episode_duration={"days": 1, "hours": 0, "minutes": 0}, time_gap={"days": 3000, "hours": 0, "minutes": 0})

env = BTgymEnv(dataset=dataset, start_cash=1000)

done = False

o = env.reset()

while not done:
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    print('ACTION: {}\nREWARD: {}\nINFO: {}'.format(action, reward, info))

env.close()
