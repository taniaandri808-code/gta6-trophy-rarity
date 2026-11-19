# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/taniaandri808-code/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/taniaandri808-code/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI PS5 trophy, five calibrated quantiles of its Exophase earn rate at a target time, scored by pinball loss on the logit scale. A logit-space growth model fits earned counts per trophy and tracked-player growth, pools slopes across trophy groups, projects to the target with a damped exponent, and widens the spread with extrapolation distance; when the recorded page history is entirely stale it widens for the un-observed Exophase refresh.

## Run it

```sh
python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR` (default `/input`) `*.in` cases and writes `$MAKEFUN_OUTPUT_DIR` (default `/output`) `<id>.out`. Python 3.13, standard library only, deterministic, no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=taniaandri808-code&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
