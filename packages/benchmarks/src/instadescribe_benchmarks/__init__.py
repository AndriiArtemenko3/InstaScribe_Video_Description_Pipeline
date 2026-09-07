"""Umbrella package for InstaDescribe evaluation benchmarks.

Each benchmark track lives in its own sibling submodule and is scored
independently. Today there is exactly one track:

- ``id_event_light_v0`` — natural-language event detection and temporal
  localization in short videos.

Future tracks (geolocation, retrieval, entity recognition, tracking, change
detection) would be added as new submodules with their own manifests, metrics
and CLIs. Nothing here shares scoring logic across tracks on purpose: a
benchmark result must be traceable to one small, readable module.
"""
