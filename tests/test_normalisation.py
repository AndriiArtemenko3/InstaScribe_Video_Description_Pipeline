"""Unit tests for the caption-templating + pronoun-grammar engine."""

import normalisation as N


def test_get_pronoun_set_known_and_fallback():
    assert N.get_pronoun_set("he") == {"subj": "he", "obj": "him", "poss": "his"}
    assert N.get_pronoun_set("THEY")["obj"] == "them"
    # Unknown or missing pronouns fall back to neutral "it".
    assert N.get_pronoun_set("dragon") == N.PRONOUN_FORMS["it"]
    assert N.get_pronoun_set(None) == N.PRONOUN_FORMS["it"]


def test_capitalize_first():
    assert N.capitalize_first("hello") == "Hello"
    assert N.capitalize_first("") == ""
    assert N.capitalize_first("a") == "A"


def test_get_first_reference_prefers_user_rename():
    renamed = {"user_renamed": True, "name": "Indy", "first_mention_label": "a man"}
    assert N.get_first_reference(renamed) == "Indy"
    auto = {"user_renamed": False, "name": "Indy", "first_mention_label": "a man"}
    assert N.get_first_reference(auto) == "a man"
    assert N.get_first_reference({}) == "someone"


def test_render_caption_template_substitutes_tokens():
    entities = {
        "char_1": {"first_mention_label": "a man", "name": "a man", "pronoun": "he"},
    }
    template = "{char_1_first} lifts {char_1_poss} hat. {char_1_subj_cap} smiles."
    assert N.render_caption_template(template, entities) == "a man lifts his hat. He smiles."


def test_render_caption_template_leaves_unknown_tokens_intact():
    template = "{char_9_first} waves and {not_a_token} stays."
    assert (
        N.render_caption_template(template, {}) == "{char_9_first} waves and {not_a_token} stays."
    )


def test_build_caption_template_replaces_longest_match_first():
    entities = {
        "char_1": {
            "first_mention_label": "older man",
            "name": "older man",
            "aliases": ["older man in a fedora"],
        },
    }
    out = N.build_caption_template("the older man in a fedora nods", ["char_1"], entities)
    assert "{char_1_first}" in out
    assert "fedora" not in out  # the longer alias was consumed whole
    assert out == "the {char_1_first} nods"


def test_apply_manual_character_rename_tracks_history():
    entities = [
        {"id": "char_1", "name": "a man", "name_history": []},
        {"id": "char_2", "name": "a woman", "name_history": []},
    ]
    out = N.apply_manual_character_rename(entities, "char_1", "Indiana")
    renamed = next(e for e in out if e["id"] == "char_1")
    assert renamed["name"] == "Indiana"
    assert renamed["user_renamed"] is True
    assert "a man" in renamed["name_history"]
    # Other entities are untouched, and the input is not mutated in place.
    assert next(e for e in out if e["id"] == "char_2")["name"] == "a woman"
    assert entities[0]["name"] == "a man"


def test_rerender_scenes_respects_locked():
    entities = [
        {"id": "char_1", "name": "Indiana", "pronoun": "he", "first_mention_label": "Indiana"}
    ]
    scenes = [
        {"caption_template": "{char_1_first} runs.", "caption": "old", "locked": False},
        {"caption_template": "{char_1_first} runs.", "caption": "old", "locked": True},
    ]
    out = N.rerender_scenes_with_updated_entities(scenes, entities)
    assert out[0]["caption"] == "Indiana runs."
    assert out[1]["caption"] == "old"  # locked scene is left alone
