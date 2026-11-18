"""
IGAD night lights and conflict dataset pipeline.

Stages (each saves its result in data/interim/ and skips itself if that
result already exists, unless --force is given):
  boundaries   second-level administrative areas and their neighbors
  lights       Black Marble monthly night lights, added up per locality
  conflict     UCDP violent events, placed in their localities
  rain         CHIRPS monthly rainfall per locality
  population   WorldPop population per locality
  build        the final locality-month dataset in data/processed/
"""
