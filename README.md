## Requirement
- Docker
- GPU (such as RTX 4060 Ti 16GB)
## Step
1. clone this repository
2. `docker compose up -d`
3. `docker compose exec python bash`
4. `python killer_whale_exec.py`
5. `python killer_whale_fine_tune.py`
6. `python killer_whale_exec_adapter.py`
7. `python killer_whale_merge.py`
8. `python killer_whale_exec_merged.py`
