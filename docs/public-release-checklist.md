# Public release checklist

Snapshot prepared on 2026-08-21. This checklist records the tree that is being made public.

## Ready

- [x] Current tracked tree contains no credential-shaped OpenRouter, GitHub, Google, or private-key value
- [x] All reachable Git commits contain no matching credential value or tracked `.env` / key file
- [x] Commit identity uses a GitHub `users.noreply.github.com` address
- [x] Runtime output (`runs/`, thoughts, summaries, local config, virtualenv, build output) is ignored
- [x] API errors redact the active key and omit raw upstream/account metadata
- [x] Native input, commercial games, and anti-cheat integration are explicitly out of scope
- [x] Documentation and replay indexes distinguish formal numbers from visual-only runs
- [x] CI workflow uses read-only repository permission and commit-pinned actions
- [x] CI passes on Ubuntu and Windows with Python 3.12
- [x] README publishes the measured PC, Cloud, and Colab environments without device/account identifiers
- [x] Manual source and full reachable-history scans find no credential, personal path, or tracked secret

## Must decide before public

- [x] Add the MIT `LICENSE`
- [x] Land the V3 and V4 pull requests so the default `main` branch contains both
- [x] Move 88 replay GIFs to experiment-specific GitHub Release assets
- [x] Enable GitHub Dependabot vulnerability alerts
- [x] Collapse the experimental PR to the final adapter tree so an intermediate upstream-derived prototype is not public history

## Security scan note

The bundled Codex Security scanner was invoked before the visibility change, but its Git path reader failed
on the Japanese workspace path with a CP932 `UnicodeDecodeError` before creating a scan. A manual fallback
reviewed the tracked tree, reachable Git patches, dependency/CI configuration, ignored runtime artifacts,
model-output execution boundary, key redaction, and GitHub alerts. No publish-blocking finding was found.
This is a tooling failure record, not a claim that an automated scan passed.

## Replay-storage decision

Replay GIFs are migrated to GitHub Release assets. Documentation keeps direct per-episode image links,
while the private Git history is rewritten before publication to remove the old 228 MiB blob archive.

GitHub retains read-only refs for closed pull requests. `refs/pull/1/head` and `refs/pull/2/head` can still
reach the pre-migration GIF blobs even after their source branches are deleted. Those blobs are replay GIFs
already intentionally public in Releases; a post-public scan found no credential or personal path in either
PR tree. The default branch/tag graph has a 122,261-byte largest blob, so normal clones remain small.

The pre-rewrite repository bundle and SHA-256 GIF manifest remain offline under the workspace artifact
directory; they are not published with the repository.

## 2026-08-24 result-hash addendum

Windowsで生成したJSONを`Get-FileHash`するとworking treeのCRLFをhashするが、このrepositoryは
`.gitattributes`でtextをLFへ正規化する。最初のColab result文書はCRLF側のSHAを書いてしまい、
GitHubから取得したblobと一致しなかった。公開値をGit LF blobのSHAへ訂正し、二つのmanual result
writerも`newline="\n"`固定へ変更した。以後、公開artifactのhashはworking fileではなくGitへ入る
LF bytesに対して検証する。
