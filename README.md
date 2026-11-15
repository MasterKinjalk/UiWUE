# UiWUE

Hackathon project on the ADA 2025 multimodal sugar beet water-deficit dataset
([paper](https://arxiv.org/html/2610.23609v1),
[dataset code](https://github.com/pippowell/ada2025)).

Goal: a tool pipeline a biologist can use to analyse water-deficit stress in
sugar beet, not only machine-learning models.

## Start here

1. Read the master plan: [plan/Master-Plan.md](plan/Master-Plan.md) (who builds what, the table contract, the deadline)
2. Set up your environment: [docs/SOFTWARE_STACK.md](docs/SOFTWARE_STACK.md)
3. Run the app: `python app/app.py` (from the repo root or `app/`; works without `pip install -e .`), then open <http://127.0.0.1:8050>
4. Before pushing a table: `python -m uiwue.core.validate` and `python -m pytest -q`
5. Challenge 3 (3D phenotyping): download data with `scripts/download_data.sh`; pipeline,
   results and the 3D canopy page (`/threed`) are described in [docs/threed/](docs/threed/); fine-tuned models in
   [docs/ML_MODELS.md](docs/ML_MODELS.md)
