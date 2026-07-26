"""
Prompt templates for Tier 4 agentic retrievers (CRAG and ReAct).

All prompts are in French — statutory-register vocabulary alignment with the
BSARD corpus. Do not embed prompts inline in crag.py or react.py; import from here.

CRAG prompts
------------
CRAG_REWRITE_PROMPT   : Incorrect path — full query rewrite
CRAG_DECOMPOSE_PROMPT : Ambiguous path — targeted sub-query decomposition

ReAct prompts (added in Tier 4.2)
----------------------------------
REACT_SYSTEM_PROMPT                       : System-level instruction with tool definitions
REACT_USER_TEMPLATE                       : Per-query user turn (question + scratchpad)
REACT_INVALID_ACTION_OBSERVATION          : v1 generic rejection (Round-1 default)
REACT_INVALID_ACTION_OBSERVATION_TEMPLATE : v2 templated rejection that echoes the
                                            malformed Action line back at the model
                                            (Round-2 Block A — flag-toggled)
REACT_FEWSHOT_TRAJECTORIES_BLOCK          : 3 in-prompt French few-shot trajectories
                                            drawn from BSARD train split
                                            (Round-2 Block A — flag-toggled)
REACT_GAP_PROMPT                          : Round-2 Block B — short pre-search
                                            facets-still-missing prompt
"""

# ---------------------------------------------------------------------------
# CRAG — Incorrect path (full query rewrite)
# ---------------------------------------------------------------------------

CRAG_REWRITE_PROMPT = """\
Vous êtes un expert en amélioration de requêtes pour un système de recherche \
d'articles de droit belge.

Requête originale : {original_query}
Requête actuelle  : {current_query}
Problème : Les articles récupérés sont hors sujet ou non pertinents.

Réécrivez la requête en utilisant une terminologie juridique belge plus précise \
ou en abordant la question sous un angle différent susceptible de mieux \
correspondre au vocabulaire des textes de loi belges.

Répondez uniquement avec la requête réécrite, sans explication.

Requête réécrite :\
"""

# ---------------------------------------------------------------------------
# CRAG — Ambiguous path (targeted sub-query)
# ---------------------------------------------------------------------------

CRAG_DECOMPOSE_PROMPT = """\
Vous êtes un expert en amélioration de requêtes pour un système de recherche \
d'articles de droit belge.

Requête originale : {original_query}
Problème : Les articles récupérés sont partiellement pertinents mais incomplets.

Reformulez la requête de manière plus spécifique en ciblant le concept juridique \
central qui semble manquer dans les résultats actuels. Répondez uniquement avec \
la requête reformulée, sans explication.

Requête reformulée :\
"""

# ---------------------------------------------------------------------------
# ReAct — System prompt (Tier 4.2, added here to keep prompts centralised)
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """\
Vous êtes un agent de recherche juridique spécialisé dans la législation belge. \
Votre tâche est de retrouver les articles de loi pertinents pour répondre à une \
question donnée.

Vous disposez des outils suivants :
- search[requête] : recherche les articles les plus pertinents pour une requête
- lookup[id] : affiche le texte complet d'un article par son identifiant numérique
- finish[done] : termine la recherche une fois que vous avez trouvé les articles pertinents

À chaque étape, raisonnez d'abord (Pensée), puis agissez (Action).
Format strict :
Pensée : <votre raisonnement>
Action : search[<requête>] | lookup[<id>] | finish[done]

Ne récapitulez pas les identifiants dans l'action finish. \
Utilisez finish[done] uniquement comme signal d'arrêt.\
"""

# ---------------------------------------------------------------------------
# ReAct — User turn template (Tier 4.2)
# ---------------------------------------------------------------------------

REACT_USER_TEMPLATE = """\
Question : {question}

{scratchpad}\
"""

# ---------------------------------------------------------------------------
# ReAct — Pre-flight gate rejection observation (Tier 4.2)
# ---------------------------------------------------------------------------

REACT_INVALID_ACTION_OBSERVATION = """\
Observation : Action invalide. Format attendu :
search[<requête>], lookup[<id numérique>], ou finish[done].
Recommencez avec une action correctement formatée.\
"""


# ---------------------------------------------------------------------------
# ReAct — Templated rejection (Tier 4.2 Round-2, Block A)
# ---------------------------------------------------------------------------
# Echoes the malformed Action line back at the model (max 200 chars) and demands
# the canonical bracket form. Empirically more effective than the generic v1
# rejection, which the model often re-emits unchanged.
# Placeholder: {malformed_line}

REACT_INVALID_ACTION_OBSERVATION_TEMPLATE = """\
Observation : Votre dernière action « {malformed_line} » n'est pas reconnue. \
Le seul format accepté est :
  Action : search[<requête>]
  Action : lookup[<id numérique>]
  Action : finish[done]
Réécrivez exactement une de ces trois lignes — pas de parenthèses, pas de \
deux-points, pas de guillemets français, pas de prose autour.\
"""


# ---------------------------------------------------------------------------
# ReAct — In-prompt few-shot trajectories (Tier 4.2 Round-2, Block A)
# ---------------------------------------------------------------------------
# Three abridged search → lookup → finish trajectories drawn from BSARD train
# questions (qids 675, 984, 1096). Train split only — test/val never touched.
# Prepended to REACT_SYSTEM_PROMPT when use_fewshot_trajectories=True.
# Each trajectory keeps Pensée/Action lines short (< 120 chars) to stay
# representative of the canonical format under max_tokens=128.

REACT_FEWSHOT_TRAJECTORIES_BLOCK = """\
Voici trois exemples de trajectoires correctes (chaque ligne respecte le \
format strict).

=== Exemple 1 ===
Question : Suis-je obligé de me présenter à l'audience devant le tribunal de police ?
Pensée : Je dois trouver les règles de comparution devant le tribunal de police.
Action : search[comparution obligatoire tribunal de police]
Observation : [1] ID=3395 — Code d'instruction criminelle, art. 152 …
Pensée : L'article 3395 semble central. Je vérifie son texte complet.
Action : lookup[3395]
Observation : ID=3395 — Texte complet : « Le prévenu peut comparaître par avocat … »
Pensée : J'ai trouvé la règle pertinente. Je termine.
Action : finish[done]

=== Exemple 2 ===
Question : Un colocataire seul peut-il mettre fin au bail officiel de colocation à Bruxelles ?
Pensée : Question sur le bail de colocation bruxellois — je cherche le pacte de colocation.
Action : search[fin bail colocation Bruxelles pacte]
Observation : [1] ID=11441 — Code bruxellois du logement, art. 261 …
Pensée : Cet article traite la sortie d'un colocataire. Je lis le texte complet.
Action : lookup[11441]
Observation : ID=11441 — Texte complet : « Tout colocataire peut donner congé … »
Pensée : Réponse trouvée — un colocataire peut sortir avec préavis. Je termine.
Action : finish[done]

=== Exemple 3 ===
Question : À quoi sert le secret professionnel ?
Pensée : Notion classique du droit pénal — je cherche la disposition générale.
Action : search[secret professionnel finalité droit pénal]
Observation : [1] ID=34714 — Code pénal, art. 458 …
Pensée : L'article 458 du Code pénal pose la règle. Je vérifie.
Action : lookup[34714]
Observation : ID=34714 — Texte complet : « Les médecins, … et toutes autres personnes … »
Pensée : J'ai la disposition de référence. Je termine.
Action : finish[done]

Maintenant, traitez la nouvelle question avec le même format strict.
"""


# ---------------------------------------------------------------------------
# ReAct — Gap-prompt sub-step (Tier 4.2 Round-2, Block B)
# ---------------------------------------------------------------------------
# Issued before each search after the first; returns a short list of facets
# missing from prior observations. The output is appended to the next
# generate_step prompt to push the agent away from rewording.
# Placeholders: {question}, {observations_so_far}

REACT_GAP_PROMPT = """\
Voici la question :
{question}

Voici les observations déjà collectées (extraits) :
{observations_so_far}

En 1 à 3 puces très courtes, indiquez les facettes juridiques encore \
manquantes pour répondre à la question. Soyez concret (notion, condition, \
exception). Pas de phrases complètes — uniquement des puces.

Facettes manquantes :\
"""
