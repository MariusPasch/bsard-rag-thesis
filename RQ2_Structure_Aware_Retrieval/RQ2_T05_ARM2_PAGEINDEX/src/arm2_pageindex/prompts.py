"""French prompt templates for Arm 2B navigation.

Every prompt enforces JSON-only output. Temperature is fixed at 0.0
in T01's ``LLMClient`` so prompts can rely on deterministic structure.

Prompts use ``{{`` / ``}}`` to escape literal braces inside JSON
exemplars; the rest are ``str.format`` placeholders. ``example_*_id``
placeholders are filled at format time with a real ID lifted from the
tree being shown to the model — this prevents the 8B model from
copying a hard-coded literal from the prompt instead of picking from
the actual list (observed on PDF 1804 / Code Civil: every chapter-
selection response contained `LAW_2_CH_TITRE_Ier_Principes` because
that example happened to look like a real Code Civil chapter under
the old doc_id numbering).
"""

LAW_SELECTION_PROMPT = """\
Vous êtes un expert en droit belge. Pour la question suivante, sélectionnez \
le ou les corpus législatifs les plus susceptibles de contenir la réponse.

Question : "{query}"

Corpus disponibles :
{laws_json}

Sélectionnez 1 à {max_laws} corpus. Si plusieurs corpus semblent pertinents, \
listez-les par ordre de pertinence décroissante.

IMPORTANT : utilisez uniquement les `node_id` figurant dans la liste \
ci-dessus. Ne recopiez pas l'exemple si l'identifiant ne figure pas dans \
la liste.

Répondez uniquement en JSON, sans texte avant ni après :
{{"selected_law_ids": ["{example_law_id}"], "reason": "..."}}"""


CHAPTER_SELECTION_PROMPT = """\
Question : "{query}"
Corpus législatif : "{law_title}"

Voici les chapitres et titres de ce corpus :
{chapters_json}

Sélectionnez 1 à {max_chapters} sections (chapitre, titre, partie, etc.) \
susceptibles de contenir la réponse. Plusieurs sections peuvent être pertinentes \
en cascade (par exemple un titre général et un chapitre spécifique).

IMPORTANT : utilisez uniquement les `node_id` figurant dans la liste \
ci-dessus. Ne recopiez pas l'exemple si l'identifiant ne figure pas \
dans la liste. Répondez avec UN SEUL objet JSON (pas un tableau \
d'objets) regroupant tous les choix dans ``selected_chapter_ids``.

Répondez uniquement en JSON, sans texte avant ni après :
{{"selected_chapter_ids": ["{example_chapter_id}"], "reason": "..."}}"""


ARTICLE_SELECTION_PROMPT = """\
Question : "{query}"
Section : "{chapter_title}"

Articles dans cette section (avec un résumé d'une phrase) :
{articles_json}

Sélectionnez les articles potentiellement pertinents pour la question. \
Préférez l'inclusion : si plusieurs articles paraissent même partiellement \
liés au sujet, listez-les tous (le tri fin est fait à l'étape suivante). \
Ne retournez une liste vide que si aucun article de cette section n'a un \
lien thématique plausible avec la question.

IMPORTANT : utilisez uniquement les `node_id` figurant dans la liste \
ci-dessus. Ne recopiez pas l'exemple. Répondez avec UN SEUL objet JSON \
(pas un tableau d'objets).

Répondez uniquement en JSON, sans texte avant ni après :
{{"selected_article_ids": ["{example_article_id}"], "reason": "..."}}"""


EVALUATE_PROMPT = """\
Question : "{query}"

Articles retenus jusqu'ici (texte intégral) :
{article_texts}

Tâches :
1. Notez chaque article de 0 à 3 selon sa pertinence pour la question :
   - 3 = répond directement à la question.
   - 2 = très pertinent (contexte essentiel pour la réponse).
   - 1 = vaguement pertinent.
   - 0 = non pertinent.
2. Identifiez les références à d'autres articles dans le texte (formules \
   du type « visé à l'article D.49 », « conformément à l'article ... », \
   « en application du chapitre ... »). Ne listez que les numéros d'article \
   réellement cités.

Si les articles déjà retenus suffisent à répondre, indiquez "sufficient". \
S'il faut consulter d'autres articles cités, indiquez "need_more" et \
listez leurs numéros (par exemple "D49", "R270bis11").

Répondez uniquement en JSON, sans texte avant ni après :
{{"status": "sufficient" | "need_more",
 "scores": {{"{example_article_id}": 3}},
 "additional_article_numbers": ["D49", "R270"],
 "reason": "..."}}"""
