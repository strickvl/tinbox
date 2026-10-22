# Publishing Tinbox to PyPI

Tinbox uses GitHub Actions trusted publishing through OpenID Connect (OIDC). The workflow builds and tests a version tag, then a separate job obtains a short-lived publishing credential from PyPI. No long-lived PyPI API token or GitHub secret is required.

## One-time setup

Create the `pypi` environment in the [repository's environment settings](https://github.com/strickvl/tinbox/settings/environments). If deployment branches/tags are restricted, allow the `main` branch for manual runs and `v*` tags for release-triggered runs. Required reviewers can be configured if manual approval is desired; they are not required by the workflow.

In [Tinbox's PyPI publishing settings](https://pypi.org/manage/project/tinbox/settings/publishing/), add a GitHub trusted publisher with these exact values:

| Field | Value |
| --- | --- |
| Owner | `strickvl` |
| Repository | `tinbox` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

The workflow filename has no directory prefix. Registration requires access to the PyPI project. See [PyPI's setup instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Releasing

Update `pyproject.toml`, `src/tinbox/__init__.py`, and `uv.lock` together, document the changes, and complete the repository's checks. Merge the release commit into `main`, then create an immutable `v<version>` tag at that commit.

Publishing a GitHub release for that tag starts `.github/workflows/publish.yml`. The workflow checks that the tag is on `main`'s history and matches both package and runtime versions, runs locked checks with Poppler, builds the wheel and source distribution, and smoke-tests both. Only the separate publishing job receives `id-token: write`.

For a build-only check, run the workflow manually from `main`, enter the existing version tag, and leave **publish** unchecked. To publish manually, select **publish**. This also supports an older tag whose source predates the workflow, such as the already prepared `v0.2.0`; do not move or recreate the tag.

`uv publish --trusted-publishing always` requires OIDC and fails if the PyPI registration does not match. Repeating a publish skips identical existing PyPI files and rejects files with different contents. It does not fall back to a stored API token. This workflow does not generate artifact attestations.

After publishing, verify both distributions on PyPI and test `uv tool upgrade tinbox` in an isolated tool directory. For users moving from 0.1.0, an older Python interpreter, or a GitHub install, verify the explicit-extra reinstall documented in the README. Finalize the GitHub release notes after the package is available and upgrades are verified.
