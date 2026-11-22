"""
action_recognition.py - Maps video model predictions to supported EPIC-KITCHENS verbs and nouns.
Handles vocabulary constraints (98 verbs, 301 nouns) and dataset-specific logic.
"""

import os
import yaml
from code.utils import setup_logger, load_json

logger = setup_logger("ActionRecognition")


class ActionRecognizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.supported_classes = load_json(cfg["dataset"]["supported_classes_path"])
        self.valid_verbs = set(self.supported_classes.get("verbs", []))
        self.valid_nouns = set(self.supported_classes.get("nouns", []))
        self.kinetics_mapping = self._load_kinetics_mapping(cfg["dataset"]["kinetics_mapping_path"])

    def _load_kinetics_mapping(self, mapping_path):
        if not os.path.isfile(mapping_path):
            logger.warning(f"Mapping file {mapping_path} not found. Using identity mapping.")
            return {}
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("mappings", {})

    def map_clip_to_action(self, top_categories, dataset="epic_kitchens_100"):
        """
        Maps top-K Kinetics predictions to supported verb and noun.
        Returns:
            verb: str
            noun: str
            score: float in [0, 1]
        """
        if not top_categories:
            return "other", "other", 0.1

        # Check if any of the top categories has a direct mapping
        for cat_name, prob in top_categories:
            cat_clean = cat_name.lower().strip()
            if cat_clean in self.kinetics_mapping:
                mapped = self.kinetics_mapping[cat_clean]
                verb = mapped.get("verb", "other")
                noun = mapped.get("noun", "other")
                if verb in self.valid_verbs and noun in self.valid_nouns:
                    return verb, noun, round(min(1.0, prob * 1.2), 2)

        # Fallback keyword matching for verbs and nouns
        top_cat, top_prob = top_categories[0]
        words = top_cat.lower().replace("-", " ").split()

        matched_verb = "other"
        for w in words:
            if w in self.valid_verbs and w != "other":
                matched_verb = w
                break

        matched_noun = "other"
        for w in words:
            if w in self.valid_nouns and w != "other":
                matched_noun = w
                break

        if dataset == "epic_kitchens_100" and matched_verb == "other":
            # For kitchen domain, reasonable default interaction verb is 'take' or 'put'
            matched_verb = "take"
            if matched_noun == "other":
                matched_noun = "food"

        score = round(max(0.2, min(1.0, top_prob)), 2)
        return matched_verb, matched_noun, score

    def process_clips(self, clips, dataset="epic_kitchens_100"):
        """
        Annotates all clips with predicted verb, noun, and action score.
        """
        for clip in clips:
            top_cats = clip.get("top_categories", [])
            verb, noun, score = self.map_clip_to_action(top_cats, dataset=dataset)
            clip["verb"] = verb
            clip["noun"] = noun
            clip["action_score"] = score

        return clips
