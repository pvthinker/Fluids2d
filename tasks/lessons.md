# Lessons learned

## git add <dir> embarque les fichiers non suivis

**Pattern d'erreur** : `git add src/fluids2d tests` dans un commit destiné à ne
contenir que des modifications a indexé 25 fichiers non suivis (brouillons
`toto.py`, `meanstate.py`, `*_fdse.py`, notebooks `*.ipynb`, dossiers
`hconv_ref/`, `horiz_longrun/`), transformant le commit de lint en un commit
de 60 fichiers / +9599 lignes.

**Règle à retenir** :
- Pour un commit de lint/correction, stager uniquement les chemins de fichiers
  **déjà suivis** modifiés, par ex.
  `git add $(git diff --name-only)` + `git add <chemins explicites>`.
- Grep/recenser les brouillons non ignorés (`git status --porcelain | grep '^??'`)
  **avant** `git commit` — ils n'ont pas à apparaître dans l'index.
- Récupération sûre si cela arrive : `git reset --soft HEAD~1` (le commit saute,
  index+worktree intacts), puis
  `git diff --cached --name-status | awk '$1=="A"{print $2}' | xargs git restore --staged --`
  pour renvoyer les fichiers ajoutés par erreur en état non suivi (contenu préservé),
  puis re-commit.

## Agiter le hook pre-commit local

Après `pre-commit install`, chaque commit exécute `ruff` ; penser à vérifier la
sortie du hook (ex. `ruff (legacy alias)...Passed`) avant de considérer le commit
comme valide, et à retrouver l'état réel avec `git log --oneline -n`.

